import os, re, json, pickle, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import torch


# print(torch.__version__)
# print(pandas.__version__)




from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import get_linear_schedule_with_warmup
warnings.filterwarnings('ignore')



DATA_PATH = "data/support_tickets.csv"
EPOCHS = 3
BATCH = 16
MAXLEN = 256
LR = 5e-6





os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)







P2N = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
N2P = {0: "Low", 1: "Medium", 2: "High", 3: "Critical"}

EXP_HRS = {"Low": 72, "Medium": 24, "High": 8, "Critical": 2}


print("loading data...")
df = pd.read_csv(DATA_PATH)
print(df.shape)
print(df["Priority_Level"].value_counts().to_dict())




def clean(text):
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s.,!?'-]", " ", text)

    return re.sub(r"\s+", " ", text)

def get_tier(email):

    free = {"gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "icloud.com",
            "aol.com", "example.com", "example.org", "example.net", "live.com"}
    if not isinstance(email, str):
        return "consumer"

    return "consumer" if email.split("@")[-1].lower() in free else "enterprise"




df["subject_clean"] = df["Ticket_Subject"].apply(clean)
df["desc_clean"] = df["Ticket_Description"].apply(clean)
df["priority_num"] = df["Priority_Level"].map(P2N)
df["res_hours"] = pd.to_numeric(df["Resolution_Time_Hours"], errors="coerce")
df["res_hours"] = df["res_hours"].fillna(df["res_hours"].median())
df["domain_tier"] = df["Customer_Email"].apply(get_tier)
df["channel"] = df["Ticket_Channel"].str.lower().str.strip()
df["text_len"] = df["desc_clean"].apply(lambda x: len(x.split()))
df["category"] = df["Issue_Category"].fillna("unknown")
df["ticket_id"] = df["Ticket_ID"].astype(str) if "Ticket_ID" in df.columns else df.index.astype(str)


# signal 1 - NLP
print("\ncomputing NLP signal...")

CRISIS = [
    "outage", "down", "breach", "data loss", "not working", "cannot access",
    "system failure", "completely broken", "revenue loss", "security", "hack",
    "compromised", "production down", "all users", "crash", "locked out",
    "data breach", "fraud", "emergency", "halted", "service unavailable"
]
ESCALATION = ["escalate", "supervisor", "manager", "legal", "lawsuit",
              "refund", "cancel subscription", "chargeback", "attorney"]
LOW_WORDS = ["minor", "suggestion", "feedback", "wondering", "general inquiry",
             "question", "how do i", "where is", "cosmetic", "font", "color",
             "when possible", "no hurry"]

def nlp_score(text, channel, tlen):
    crisis = sum(1 for w in CRISIS if w in text)
    esc = sum(1 for w in ESCALATION if w in text)
    low = sum(1 for w in LOW_WORDS if w in text)

    kw = max(0.0, min(1.0, crisis * 0.30 + esc * 0.25 - low * 0.15))
    lc = min(1.0, tlen / 80.0)




    ch = {"chat": 0.20, "web form": 0.10, "email": 0.0}.get(channel, 0.0)
    score = 0.50 * kw + 0.35 * lc + 0.15 * ch


    if score >= 0.55:
        return 3
    elif score >= 0.35:
        return 2
    elif score >= 0.18:
        return 1
    return 0

df["nlp_sev"] = df.apply(lambda r: nlp_score(r["desc_clean"], r["channel"], r["text_len"]), axis=1)
print(df["nlp_sev"].value_counts().sort_index().to_dict())


# signal 2 - resolution time anomaly
print("\ncomputing resolution time signal...")

res_bounds = {}




def res_signal(df):
    sev = df["priority_num"].copy().astype(int)
    for priority, pnum in P2N.items():
        mask = df["Priority_Level"] == priority
        group = df.loc[mask, "res_hours"]
        p15 = group.quantile(0.15)



        p85 = group.quantile(0.85)
        res_bounds[priority] = (float(p15), float(p85))
        sev.loc[mask & (df["res_hours"] <= p15)] = min(3, pnum + 2)


        sev.loc[mask & (df["res_hours"] >= p85)] = max(0, pnum - 2)
    return sev

df["reg_sev"] = res_signal(df)
print(df["reg_sev"].value_counts().sort_index().to_dict())

with open("models/res_bounds.pkl", "wb") as f:
    pickle.dump(res_bounds, f)


# fuse
print("\nfusing signals and generating labels...")

df["inferred_num"] = (0.4 * df["nlp_sev"] + 0.6 * df["reg_sev"]).round().clip(0, 3).astype(int)
df["inferred"] = df["inferred_num"].map(N2P)
df["delta"] = df["inferred_num"] - df["priority_num"]
df["gap"] = df["delta"].abs()
df["mismatch"] = (df["gap"] >= 1).astype(int)

def get_mtype(row):
    if row["mismatch"] == 0:


        return "Consistent"

    return "Hidden Crisis" if row["delta"] > 0 else "False Alarm"

df["mismatch_type"] = df.apply(get_mtype, axis=1)

n = len(df)
nm = df["mismatch"].sum()
print(f"match: {n-nm}  mismatch: {nm}")
print(df["mismatch_type"].value_counts().to_dict())

# ablation stats
s1 = (df["nlp_sev"] == df["priority_num"]).mean()


s2 = (df["reg_sev"] == df["priority_num"]).mean()
pair = (df["nlp_sev"] == df["reg_sev"]).mean()
print(f"\nablation — nlp: {s1:.3f}  resolution: {s2:.3f}  pairwise: {pair:.3f}")

df.to_csv("outputs/pseudo_labels.csv", index=False)




# build classifier input
def res_bin(h):
    if h < 12:
        return "under 12h"

    elif h < 36:
        return "12 to 36h"
    elif h < 72:
        return "36 to 72h"
    elif h < 120:
        return "72 to 120h"
    return "over 120h"

df["clf_input"] = (
    "Subject: " + df["subject_clean"] + " . " +
    "Details: " + df["desc_clean"] + " . " +
    "Priority: " + df["Priority_Level"] + " . " +
    "Resolution: " + df["res_hours"].apply(res_bin) + " . " +
    "Channel: " + df["channel"] + " . " +
    "Category: " + df["category"] + " . " +
    "Tier: " + df["domain_tier"]
)


# train
print("\ntraining...")

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")

else:
    device = torch.device("cpu")
print(f"device: {device}")

texts = df["clf_input"].tolist()
labels = df["mismatch"].tolist()

X_tr, X_te, y_tr, y_te = train_test_split(texts, labels, test_size=0.15, random_state=42, stratify=labels)
X_tr, X_va, y_tr, y_va = train_test_split(X_tr, y_tr, test_size=0.15, random_state=42, stratify=y_tr)


print(f"train: {len(X_tr)}  val: {len(X_va)}  test: {len(X_te)}")

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")







class TicketDS(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, i):
        enc = tokenizer(self.texts[i], max_length=MAXLEN, padding="max_length",
                        truncation=True, return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[i], dtype=torch.long)
        }

cc = np.bincount(y_tr)
sample_weights = [1.0 / cc[l] for l in y_tr]
sampler = WeightedRandomSampler(sample_weights, len(y_tr), replacement=True)

tr_dl = DataLoader(TicketDS(X_tr, y_tr), batch_size=BATCH, sampler=sampler)
va_dl = DataLoader(TicketDS(X_va, y_va), batch_size=BATCH, shuffle=False)


te_dl = DataLoader(TicketDS(X_te, y_te), batch_size=BATCH, shuffle=False)

model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
model.to(device)

for name, param in model.named_parameters():
    if not any(x in name for x in ["layer.4", "layer.5", "pre_classifier", "classifier"]):
        param.requires_grad = False




cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_tr)
criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float).to(device))
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=0.01
)
total_steps = len(tr_dl) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps)


def evaluate(loader):
    model.eval()
    preds, true = [], []
    with torch.no_grad():
        for b in loader:
            out = model(input_ids=b["input_ids"].to(device),
                        attention_mask=b["attention_mask"].to(device))
            preds.extend(out.logits.argmax(-1).cpu().tolist())
            true.extend(b["label"].tolist())
    return preds, true


best_f1 = 0
best_weights = None





for epoch in range(EPOCHS):
    model.train()
    running = 0


    for i, b in enumerate(tr_dl):
        optimizer.zero_grad()
        out = model(input_ids=b["input_ids"].to(device),
                    attention_mask=b["attention_mask"].to(device))
        loss = criterion(out.logits, b["label"].to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()



        scheduler.step()
        running += loss.item()
        if (i + 1) % 100 == 0:
            print(f"epoch {epoch+1} step {i+1}/{len(tr_dl)} loss {running/(i+1):.4f}")

    preds, true = evaluate(va_dl)
    f1 = f1_score(true, preds, average="macro")
    acc = accuracy_score(true, preds)


    print(f"epoch {epoch+1} — val acc: {acc:.4f}  f1: {f1:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

model.load_state_dict(best_weights)
model.to(device)



preds, true = evaluate(te_dl)
print("\nresults:")
print(classification_report(true, preds, target_names=["Consistent", "Mismatch"]))

model.save_pretrained("models/distilbert_mismatch")
tokenizer.save_pretrained("models/distilbert_mismatch")
print("model saved")





# dossiers
print("\ngenerating dossiers...")

CRISIS_SHORT = ["outage", "down", "breach", "not working", "cannot access",
                "crash", "locked out", "compromised", "fraud", "emergency", "data loss"]

def make_analysis(info):
    try:
        import requests
        r = requests.post("http://localhost:11434/api/generate",
                          json={"model": "mistral", "stream": False,
                                "prompt": (
                                    "You are a support ticket auditor. Write 2-3 sentences "
                                    "explaining why this ticket has a priority mismatch. "
                                    "Only use these facts and nothing else:\n"
                                    f"Subject: {info['subject']}\n"
                                    f"Description: {info['desc']}\n"
                                    f"Channel: {info['channel']}\n"
                                    f"Resolution time: {info['res_hours']:.0f} hours\n"
                                    f"Assigned: {info['assigned']}\n"
                                    f"Inferred: {info['inferred']}")},
                          timeout=30)
        if r.status_code == 200:
            return r.json()["response"].strip()



    except:
        pass
    direction = "underestimated" if info["delta"] > 0 else "overestimated"
    exp = EXP_HRS.get(info["assigned"], 24)
    return (f"The assigned priority '{info['assigned']}' appears {direction}. "
            f"Resolution took {info['res_hours']:.0f}h (expected ~{exp}h for "
            f"'{info['assigned']}' tickets) via {info['channel']}.")

model.eval()




all_preds, all_confs = [], []
with torch.no_grad():
    for i in range(0, len(X_te), BATCH):
        enc = tokenizer(X_te[i:i+BATCH], max_length=MAXLEN, padding=True,
                        truncation=True, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)
        probs = torch.softmax(out.logits, -1)
        all_preds.extend(out.logits.argmax(-1).cpu().tolist())


        all_confs.extend(probs[:, 1].cpu().tolist())

test_df = df[df["clf_input"].isin(set(X_te))].iloc[:len(X_te)].reset_index(drop=True)

dossiers = []
for i, (pred, conf) in enumerate(zip(all_preds, all_confs)):
    if pred != 1 or i >= len(test_df):
        continue
    if len(dossiers) >= 20:
        break

    row = test_df.iloc[i]
    text = row["subject_clean"] + " " + row["desc_clean"]



    rt = float(row["res_hours"])
    exp = EXP_HRS.get(row["Priority_Level"], 24)
    ratio = rt / max(exp, 0.1)





    if ratio > 2:
        interp = f"{rt:.0f}h is {ratio:.1f}x above expected {exp}h for {row['Priority_Level']}"
    elif ratio < 0.4:
        interp = f"{rt:.0f}h is much faster than expected {exp}h for {row['Priority_Level']}"
    else:
        interp = f"{rt:.0f}h is within normal range for {row['Priority_Level']}"



    kw_ev = [
        {"signal": "keyword", "value": kw,
         "weight": str(round(min(0.9, 0.3 + text.count(kw) * 0.15), 2))}
        for kw in CRISIS_SHORT if kw in text
    ][:3]



    dossiers.append({
        "ticket_id": row["ticket_id"],
        "assigned_priority": row["Priority_Level"],
        "inferred_severity": row["inferred"],
        "mismatch_type": row["mismatch_type"],
        "severity_delta": f"{int(row['delta']):+d}",
        "feature_evidence": kw_ev + [
            {"signal": "resolution_time", "value": f"{rt:.0f}h", "interpretation": interp},
            {"signal": "channel", "value": row["channel"],
             "interpretation": f"submitted via {row['channel']}"}
        ],
        "constraint_analysis": make_analysis({
            "subject": row["subject_clean"],
            "desc": row["desc_clean"],
            "channel": row["channel"],
            "res_hours": rt,
            "assigned": row["Priority_Level"],
            "inferred": row["inferred"],
            "delta": int(row["delta"])
        }),
        "confidence": str(round(conf, 3))
    })



    

with open("outputs/dossiers.json", "w") as f:
    json.dump(dossiers, f, indent=2)

print(f"done. {len(dossiers)} dossiers saved.")

"""
predict.py
usage: python predict.py --input your_tickets.csv --output results.csv
"""
import argparse
import os
import re
import json


import pickle
import numpy as np

# print(np.__version__)

import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

P2N = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}




N2P = {0: "Low", 1: "Medium", 2: "High", 3: "Critical"}
EXP_HRS = {"Low": 72, "Medium": 24, "High": 8, "Critical": 2}

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


def clean(text):
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s.,!?'-]", " ", text.lower().strip()))

def get_tier(email):
    free = {"gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "icloud.com",
            "aol.com", "example.com", "example.org", "example.net"}
    if not isinstance(email, str):
        return "consumer"
    return "consumer" if email.split("@")[-1].lower() in free else "enterprise"

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

def get_inferred(row, res_bounds):
    priority = row["Priority_Level"]
    pnum = P2N.get(priority, 0)
    res = row["res_hours"]
    lo, hi = res_bounds.get(priority, (24.0, 72.0))
    if res <= lo:
        sev = min(3, pnum + 2)
    elif res >= hi:
        sev = max(0, pnum - 2)
    else:
        sev = pnum
    nlp = nlp_score(row["desc_clean"], row["channel"], row["text_len"])
    inferred = round(0.4 * nlp + 0.6 * sev)
    return max(0, min(3, inferred))





def make_analysis(info):
    try:
        import requests
        r = requests.post("http://localhost:11434/api/generate",
                          json={"model": "mistral", "stream": False,
                                "prompt": (
                                    "Write 2-3 sentences explaining why this support ticket "
                                    "has a priority mismatch. Only use these facts:\n"
                                    f"Subject: {info['subject']}\n"
                                    f"Description: {info['desc']}\n"
                                    f"Channel: {info['channel']}\n"
                                    f"Resolution: {info['res_hours']:.0f}h\n"
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
            f"Resolution took {info['res_hours']:.0f}h (expected ~{exp}h) via {info['channel']}.")






def predict(input_path, output_path):
    if not os.path.exists("models/distilbert_mismatch"):
        print("model not found. run train_pipeline.py first.")
        return

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    tokenizer = DistilBertTokenizer.from_pretrained("models/distilbert_mismatch")
    model = DistilBertForSequenceClassification.from_pretrained("models/distilbert_mismatch")
    model.to(device)
    model.eval()

    with open("models/res_bounds.pkl", "rb") as f:
        res_bounds = pickle.load(f)




    df = pd.read_csv(input_path)
    df["subject_clean"] = df["Ticket_Subject"].apply(clean)
    df["desc_clean"] = df["Ticket_Description"].apply(clean)
    df["res_hours"] = pd.to_numeric(df["Resolution_Time_Hours"], errors="coerce").fillna(24)
    df["channel"] = df["Ticket_Channel"].str.lower().str.strip()
    df["domain_tier"] = df["Customer_Email"].apply(get_tier)
    df["category"] = df["Issue_Category"].fillna("unknown")
    df["text_len"] = df["desc_clean"].apply(lambda x: len(x.split()))
    df["ticket_id"] = df["Ticket_ID"].astype(str) if "Ticket_ID" in df.columns else df.index.astype(str)

    df["inferred_num"] = df.apply(lambda r: get_inferred(r, res_bounds), axis=1)
    df["inferred"] = df["inferred_num"].map(N2P)
    df["priority_num"] = df["Priority_Level"].map(P2N)
    df["delta"] = df["inferred_num"] - df["priority_num"]




    df["clf_input"] = (
        "Subject: " + df["subject_clean"] + " . " +
        "Details: " + df["desc_clean"] + " . " +
        "Priority: " + df["Priority_Level"] + " . " +
        "Resolution: " + df["res_hours"].apply(res_bin) + " . " +
        "Channel: " + df["channel"] + " . " +
        "Category: " + df["category"] + " . " +
        "Tier: " + df["domain_tier"]
    )

    texts = df["clf_input"].tolist()
    preds, confs = [], []

    with torch.no_grad():
        for i in range(0, len(texts), 32):
            enc = tokenizer(texts[i:i+32], max_length=256, padding=True,
                            truncation=True, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            probs = torch.softmax(out.logits, -1)



            preds.extend(out.logits.argmax(-1).cpu().tolist())
            confs.extend(probs[:, 1].cpu().tolist())

    df["predicted_mismatch"] = preds
    df["confidence"] = [round(c, 3) for c in confs]
    df["result"] = df["predicted_mismatch"].map({0: "Consistent", 1: "Mismatch"})




    def mtype(row):
        if row["predicted_mismatch"] == 0:
            return "Consistent"
        return "Hidden Crisis" if row["delta"] > 0 else "False Alarm"
    df["mismatch_type"] = df.apply(mtype, axis=1)

    CRISIS_SHORT = ["outage", "down", "breach", "not working", "cannot access",
                    "crash", "locked out", "compromised", "fraud", "emergency"]



    dossiers = []
    for _, row in df[df["predicted_mismatch"] == 1].iterrows():
        text = row["subject_clean"] + " " + row["desc_clean"]
        rt = float(row["res_hours"])
        exp = EXP_HRS.get(row["Priority_Level"], 24)
        ratio = rt / max(exp, 0.1)



        if ratio > 2:
            interp = f"{rt:.0f}h is {ratio:.1f}x above expected {exp}h"
        elif ratio < 0.4:
            interp = f"{rt:.0f}h is much faster than expected {exp}h"
        else:
            interp = f"{rt:.0f}h within normal range"

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
                 "interpretation": f"via {row['channel']}"}
            ],
            "constraint_analysis": make_analysis({
                "subject": row["subject_clean"], "desc": row["desc_clean"],
                "channel": row["channel"], "res_hours": rt,
                "assigned": row["Priority_Level"], "inferred": row["inferred"],
                "delta": int(row["delta"])
            }),
            "confidence": str(round(row["confidence"], 3))
        })




    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    out_cols = ["ticket_id", "Priority_Level", "inferred", "result", "mismatch_type", "delta", "confidence"]
    df[out_cols].to_csv(output_path, index=False)

    dossier_path = output_path.replace(".csv", "_dossiers.json")
    with open(dossier_path, "w") as f:
        json.dump(dossiers, f, indent=2)

    print(f"predictions saved to {output_path}")



    print(f"dossiers saved to {dossier_path}")
    print(f"mismatches: {df['predicted_mismatch'].sum()} / {len(df)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()


    parser.add_argument("--input", default="data/support_tickets.csv")
    parser.add_argument("--output", default="outputs/predictions.csv")
    args = parser.parse_args()
    predict(args.input, args.output)

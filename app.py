import os, re, json, pickle, warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
warnings.filterwarnings('ignore')


# def download_model_if_needed():
#     if not os.path.exists("models/distilbert_mismatch"):
#         from huggingface_hub import snapshot_download
#         os.makedirs("models", exist_ok=True)
#         snapshot_download(
#             repo_id="aryansharma72062/sia-distilbert",
#             local_dir="models/distilbert_mismatch",
#             ignore_patterns=["res_bounds.pkl"]
#         )
#     if not os.path.exists("models/res_bounds.pkl"):
#         from huggingface_hub import hf_hub_download
#         os.makedirs("models", exist_ok=True)
#         hf_hub_download(
#             repo_id="aryansharma72062/sia-distilbert",
#             filename="res_bounds.pkl",
#             local_dir="models"
#         )

# download_model_if_needed()




st.set_page_config(page_title="Support Integrity Auditor", page_icon="🔍", layout="wide")

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
CRISIS_SHORT = ["outage", "down", "breach", "not working", "cannot access",
                "crash", "locked out", "compromised", "fraud", "emergency", "data loss"]


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
    if h < 12:   return "under 12h"
    elif h < 36: return "12 to 36h"
    elif h < 72: return "36 to 72h"
    elif h < 120:return "72 to 120h"
    return "over 120h"

def nlp_score(text, channel, tlen):
    crisis = sum(1 for w in CRISIS if w in text)
    esc = sum(1 for w in ESCALATION if w in text)
    low = sum(1 for w in LOW_WORDS if w in text)
    kw = max(0.0, min(1.0, crisis * 0.30 + esc * 0.25 - low * 0.15))
    lc = min(1.0, tlen / 80.0)
    ch = {"chat": 0.20, "web form": 0.10, "email": 0.0}.get(channel, 0.0)
    score = 0.50 * kw + 0.35 * lc + 0.15 * ch
    if score >= 0.55:   return 3
    elif score >= 0.35: return 2
    elif score >= 0.18: return 1
    return 0

def get_inferred(res, priority, tlen, text, channel, res_bounds):
    pnum = P2N.get(priority, 0)
    lo, hi = res_bounds.get(priority, (24.0, 72.0))
    if res <= lo:
        reg = min(3, pnum + 2)
    elif res >= hi:
        reg = max(0, pnum - 2)
    else:
        reg = pnum
    nlp = nlp_score(text, channel, tlen)
    return max(0, min(3, round(0.4 * nlp + 0.6 * reg)))


@st.cache_resource
def load_model():
    
    dev = (torch.device("mps") if torch.backends.mps.is_available()
           else torch.device("cuda") if torch.cuda.is_available()
           else torch.device("cpu"))
    tok = DistilBertTokenizer.from_pretrained("aryansharma72062/sia-distilbert")
    mdl = DistilBertForSequenceClassification.from_pretrained("aryansharma72062/sia-distilbert")
    mdl.to(dev)
    mdl.eval()
    bounds = {}
    from huggingface_hub import hf_hub_download
    bounds_path = hf_hub_download(repo_id="aryansharma72062/sia-distilbert", filename="res_bounds.pkl")
    with open(bounds_path, "rb") as f:
        bounds = pickle.load(f)
    return mdl, tok, dev, bounds


def predict_one(ticket, mdl, tok, dev, bounds):
    subj = clean(ticket.get("Ticket_Subject", ""))
    desc = clean(ticket.get("Ticket_Description", ""))
    priority = ticket.get("Priority_Level", "Low")
    channel = ticket.get("Ticket_Channel", "email").lower().strip()
    tier = get_tier(ticket.get("Customer_Email", ""))
    res = float(ticket.get("Resolution_Time_Hours", 24))
    category = ticket.get("Issue_Category", "unknown")
    tlen = len(desc.split())
    text = subj + " " + desc

    inferred_num = get_inferred(res, priority, tlen, text, channel, bounds)
    inferred = N2P[inferred_num]
    delta = inferred_num - P2N.get(priority, 0)

    clf_input = (f"Subject: {subj} . Details: {desc} . Priority: {priority} . "
                 f"Resolution: {res_bin(res)} . Channel: {channel} . "
                 f"Category: {category} . Tier: {tier}")

    enc = tok(clf_input, max_length=256, padding=True, truncation=True, return_tensors="pt")
    enc = {k: v.to(dev) for k, v in enc.items()}
    with torch.no_grad():
        out = mdl(**enc)
        probs = torch.softmax(out.logits, -1)
        pred = out.logits.argmax(-1).item()
        conf = probs[0, 1].item()

    mtype = "Consistent"
    if pred == 1:
        mtype = "Hidden Crisis" if delta > 0 else "False Alarm"

    return {
        "pred": pred, "conf": round(conf, 3),
        "assigned": priority, "inferred": inferred,
        "delta": delta, "mtype": mtype,
        "res": res, "channel": channel,
        "subject": subj, "desc": desc
    }


def make_dossier(r):
    text = r["subject"] + " " + r["desc"]
    rt = r["res"]
    exp = EXP_HRS.get(r["assigned"], 24)
    ratio = rt / max(exp, 0.1)

    if ratio > 2:
        interp = f"{rt:.0f}h is {ratio:.1f}x above expected {exp}h for {r['assigned']}"
    elif ratio < 0.4:
        interp = f"{rt:.0f}h is much faster than expected {exp}h for {r['assigned']}"
    else:
        interp = f"{rt:.0f}h is within normal range for {r['assigned']}"

    kw_ev = [
        {"signal": "keyword", "value": kw,
         "weight": str(round(min(0.9, 0.3 + text.count(kw) * 0.15), 2))}
        for kw in CRISIS_SHORT if kw in text
    ][:3]

    analysis = get_analysis(r)

    return {
        "assigned_priority": r["assigned"],
        "inferred_severity": r["inferred"],
        "mismatch_type": r["mtype"],
        "severity_delta": f"{r['delta']:+d}",
        "feature_evidence": kw_ev + [
            {"signal": "resolution_time", "value": f"{rt:.0f}h", "interpretation": interp},
            {"signal": "channel", "value": r["channel"],
             "interpretation": f"submitted via {r['channel']}"}
        ],
        "constraint_analysis": analysis,
        "confidence": str(r["conf"])
    }


def get_analysis(r):
    try:
        import requests
        resp = requests.post("http://localhost:11434/api/generate",
                             json={"model": "mistral", "stream": False,
                                   "prompt": (
                                       "You are a support ticket auditor. Write 2-3 sentences "
                                       "explaining why this ticket has a priority mismatch. "
                                       "Only use these facts and nothing else:\n"
                                       f"Subject: {r['subject']}\n"
                                       f"Description: {r['desc']}\n"
                                       f"Channel: {r['channel']}\n"
                                       f"Resolution: {r['res']:.0f}h\n"
                                       f"Assigned: {r['assigned']}\n"
                                       f"Inferred: {r['inferred']}")},
                             timeout=30)
        if resp.status_code == 200:
            return resp.json()["response"].strip()
    except:
        pass
    direction = "underestimated" if r["delta"] > 0 else "overestimated"
    exp = EXP_HRS.get(r["assigned"], 24)
    return (f"The assigned priority '{r['assigned']}' appears {direction}. "
            f"Resolution took {r['res']:.0f}h (expected ~{exp}h for "
            f"'{r['assigned']}' tickets) via {r['channel']}.")


# ── UI ────────────────────────────────────────────────────────────────────────

st.title("🔍 Support Integrity Auditor")
st.caption("Detects priority mismatches in CRM support tickets")

mdl, tok, dev, bounds = load_model()

if mdl is None:
    st.error("Model not found. Run train_pipeline.py first.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["Single Ticket", "Batch Upload", "Dashboard"])


# ── Tab 1: Single ticket ──────────────────────────────────────────────────────
with tab1:
    st.subheader("Analyze a ticket")

    with st.form("form"):
        c1, c2 = st.columns(2)
        with c1:
            subj = st.text_input("Subject", placeholder="e.g. Login not working")
            priority = st.selectbox("Assigned Priority", ["Low", "Medium", "High", "Critical"])
            channel = st.selectbox("Channel", ["Email", "Chat", "Web Form"])
        with c2:
            category = st.selectbox("Category", ["Technical", "Billing", "Account", "General Inquiry", "Fraud"])
            email = st.text_input("Customer Email", placeholder="user@company.com")
            res_time = st.text_input("Resolution Time (hours)", placeholder="48")

        desc = st.text_area("Description", height=100, placeholder="Describe the issue...")
        go = st.form_submit_button("Analyze", use_container_width=True)

    if go:
        if not subj or not desc:
            st.warning("Subject and Description are required.")
        else:
            ticket = {
                "Ticket_Subject": subj,
                "Ticket_Description": desc,
                "Priority_Level": priority,
                "Ticket_Channel": channel,
                "Issue_Category": category,
                "Customer_Email": email or "user@gmail.com",
                "Resolution_Time_Hours": res_time or "24"
            }
            with st.spinner("analyzing..."):
                r = predict_one(ticket, mdl, tok, dev, bounds)

            if r["pred"] == 0:
                st.success(f"✅ Consistent — priority looks correct  (confidence: {r['conf']:.0%})")
            else:
                st.error(f"🚨 Mismatch detected  (confidence: {r['conf']:.0%})")
                with st.expander("Evidence Dossier", expanded=True):
                    d = make_dossier(r)
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Assigned", d["assigned_priority"])
                    c2.metric("Inferred", d["inferred_severity"])
                    c3.metric("Type", d["mismatch_type"])

                    st.markdown("**Evidence**")
                    for ev in d["feature_evidence"]:
                        sig = ev["signal"]
                        val = ev["value"]
                        extra = ev.get("weight", ev.get("interpretation", ""))
                        st.markdown(f"- `{sig}`: {val} — {extra}")

                    st.markdown("**Analysis**")
                    st.info(d["constraint_analysis"])

                    st.markdown("**Full dossier (JSON)**")
                    st.json(d)


# ── Tab 2: Batch upload ───────────────────────────────────────────────────────
with tab2:
    st.subheader("Batch analysis")
    st.caption("Upload a CSV with the same columns as the training data.")

    f = st.file_uploader("Upload CSV", type=["csv"])

    if f:
        df_up = pd.read_csv(f)
        st.write(f"{len(df_up)} tickets loaded")
        st.dataframe(df_up.head(3), use_container_width=True)

        if st.button("Run analysis", use_container_width=True):
            results = []
            prog = st.progress(0)
            for i, row in df_up.iterrows():
                r = predict_one(row.to_dict(), mdl, tok, dev, bounds)
                results.append({
                    "ticket_id":      row.get("Ticket_ID", i),
                    "Priority_Level": r["assigned"],
                    "assigned":       r["assigned"],
                    "inferred":       r["inferred"],
                    "result":         "Mismatch" if r["pred"] == 1 else "Consistent",
                    "mismatch_type":  r["mtype"],
                    "confidence":     r["conf"],
                    "delta":          r["delta"],
                    "Issue_Category": row.get("Issue_Category", "unknown"),
                    "Ticket_Channel": row.get("Ticket_Channel", "unknown")
                })
                prog.progress((i + 1) / len(df_up))

            st.session_state["batch"] = pd.DataFrame(results)
            prog.empty()

        if "batch" in st.session_state:
            df_res = st.session_state["batch"]
            n_mis = (df_res["result"] == "Mismatch").sum()
            c1, c2, c3 = st.columns(3)
            c1.metric("Total", len(df_res))
            c2.metric("Mismatches", n_mis)
            c3.metric("Hidden Crises", (df_res["mismatch_type"] == "Hidden Crisis").sum())

            st.dataframe(df_res, use_container_width=True)
            st.download_button("Download results", df_res.to_csv(index=False),
                               "predictions.csv", "text/csv")


# ── Tab 3: Dashboard ──────────────────────────────────────────────────────────
with tab3:
    st.subheader("Priority Mismatch Dashboard")

    # use batch results if available, else load pseudo_labels
    if "batch" in st.session_state:
        df_dash = st.session_state["batch"]
        mis_col = "result"
        type_col = "mismatch_type"
        is_batch = True
    elif os.path.exists("outputs/pseudo_labels.csv"):
        df_dash = pd.read_csv("outputs/pseudo_labels.csv")
        mis_col = "mismatch"
        type_col = "mismatch_type"
        is_batch = False
        st.caption("Showing training data. Upload a CSV in the Batch tab to see live predictions.")
    else:
        st.info("Run training first or upload tickets in the Batch tab.")
        st.stop()

    total = len(df_dash)
    if is_batch:
        n_mis = (df_dash[mis_col] == "Mismatch").sum()
    else:
        n_mis = int(df_dash[mis_col].sum())

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total tickets", total)
    k2.metric("Mismatches", n_mis)
    k3.metric("Hidden Crises", int((df_dash[type_col] == "Hidden Crisis").sum()))
    k4.metric("False Alarms", int((df_dash[type_col] == "False Alarm").sum()))

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Mismatch vs Consistent**")
        pie = pd.DataFrame({
            "label": ["Consistent", "Mismatch"],
            "count": [total - n_mis, n_mis]
        })
        fig = px.pie(pie, names="label", values="count", hole=0.4,
                     color_discrete_sequence=["#22c55e", "#ef4444"])
        fig.update_layout(margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Mismatch Types**")
        tc = df_dash[type_col].value_counts().reset_index()
        tc.columns = ["type", "count"]
        tc = tc[tc["type"] != "Consistent"]
        fig2 = px.bar(tc, x="type", y="count",
                      color_discrete_sequence=["#f97316"])
        fig2.update_layout(showlegend=False, margin=dict(t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    # mismatch by channel
    col3, col4 = st.columns(2)

    with col3:
        ch_col = "Ticket_Channel" if "Ticket_Channel" in df_dash.columns else "channel"
        if ch_col in df_dash.columns:
            st.markdown("**Mismatches by Channel**")
            if is_batch:
                ch_data = df_dash[df_dash[mis_col] == "Mismatch"].groupby(ch_col).size().reset_index(name="count")
            else:
                ch_data = df_dash[df_dash[mis_col] == 1].groupby(ch_col).size().reset_index(name="count")
            fig3 = px.bar(ch_data, x=ch_col, y="count",
                          color_discrete_sequence=["#3b82f6"])
            fig3.update_layout(margin=dict(t=10, b=10))
            st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown("**Assigned vs Inferred Priority**")
        if "Priority_Level" in df_dash.columns and "inferred" in df_dash.columns:
            order = ["Low", "Medium", "High", "Critical"]
            a = df_dash["Priority_Level"].value_counts().reindex(order).fillna(0).reset_index()
            a.columns = ["priority", "count"]; a["type"] = "Assigned"
            b = df_dash["inferred"].value_counts().reindex(order).fillna(0).reset_index()
            b.columns = ["priority", "count"]; b["type"] = "Inferred"
            fig4 = px.bar(pd.concat([a, b]), x="priority", y="count",
                          color="type", barmode="group",
                          category_orders={"priority": order},
                          color_discrete_map={"Assigned": "#94a3b8", "Inferred": "#f97316"})
            fig4.update_layout(margin=dict(t=10, b=10))
            st.plotly_chart(fig4, use_container_width=True)

    # severity delta heatmap
    st.markdown("**Severity Delta Heatmap — Category × Channel**")
    st.caption("Average gap between inferred and assigned severity. Positive = underrated.")

    cat_col = "Issue_Category" if "Issue_Category" in df_dash.columns else "category"
    ch_col2 = "Ticket_Channel" if "Ticket_Channel" in df_dash.columns else "channel"
    d_col = "delta" if "delta" in df_dash.columns else None

    if cat_col in df_dash.columns and ch_col2 in df_dash.columns and d_col:
        hm = (df_dash.groupby([cat_col, ch_col2])[d_col]
              .mean().reset_index()
              .pivot(index=cat_col, columns=ch_col2, values=d_col)
              .fillna(0))
        fig5 = px.imshow(hm, text_auto=".1f", aspect="auto",
                         color_continuous_scale="RdYlGn_r",
                         labels={"color": "avg delta"})
        fig5.update_layout(margin=dict(t=10, b=10))
        st.plotly_chart(fig5, use_container_width=True)
    else:
        st.info("Upload batch data with category and channel columns to see heatmap.")

import html
import time
from datetime import datetime

import pandas as pd
import streamlit as st

from engine import (
    analyze, read_pdf, second_opinion, second_opinion_all, agreement,
    has_key, MAX_CLAIMS, ERRORS,
)

st.set_page_config(page_title="CiteVerify", page_icon="🔎", layout="wide")

VERDICT_STYLE = {
    "Supported": ("#16a34a", "#dcfce7", "✅"),
    "Partially supported": ("#d97706", "#fef3c7", "🟡"),
    "Unsupported": ("#dc2626", "#fee2e2", "❌"),
    "Contradictory": ("#b91c1c", "#fee2e2", "⛔"),
    "Needs human review": ("#475569", "#e2e8f0", "👀"),
}
VERDICT_ORDER = ["Supported", "Partially supported", "Unsupported",
                 "Contradictory", "Needs human review"]

st.markdown(
    """
<style>
.block-container {padding-top: 1.6rem; max-width: 1200px;}
.cv-hero {background: linear-gradient(120deg,#0f172a 0%,#1e3a8a 60%,#2563eb 100%);
  border-radius: 16px; padding: 28px 32px; color: #fff; margin-bottom: 18px;}
.cv-hero h1 {color:#fff; margin:0; font-size: 2.2rem;}
.cv-hero p {color:#dbeafe; margin:6px 0 0 0; font-size: 1.02rem;}
.cv-pills {margin-top: 14px;}
.cv-pill {display:inline-block; background:rgba(255,255,255,.14); color:#fff;
  border-radius: 999px; padding: 4px 12px; margin: 0 8px 6px 0; font-size: .82rem;}
.cv-badge {display:inline-block; border-radius: 999px; padding: 3px 12px;
  font-weight: 600; font-size: .85rem;}
.cv-card {border-radius: 12px; padding: 14px 16px; border: 1px solid #e2e8f0;
  background: #ffffff; margin-bottom: 8px; color: #0f172a;}
.cv-card .lbl {font-size:.78rem; color:#64748b; text-transform: uppercase; letter-spacing:.04em;}
.cv-card .val {font-size:1.7rem; font-weight:700; line-height:1.2;}
.cv-box {border-left: 5px solid #94a3b8; background:#f8fafc; border-radius: 8px;
  padding: 10px 14px; margin: 8px 0; color:#0f172a;}
div.stButton > button[kind="primary"] {border-radius: 10px; padding: .55rem 1.4rem; font-weight:600;}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="cv-hero">
  <h1>🔎 CiteVerify</h1>
  <p>Does the cited paper really say that? Check every claim against its reference, in seconds.</p>
  <div class="cv-pills">
    <span class="cv-pill">Finds fake or wrong references</span>
    <span class="cv-pill">Quote-checked evidence</span>
    <span class="cv-pill">Two independent AI models</span>
    <span class="cv-pill">Human review with audit trail</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)
st.caption("Results are indicators to support human review, not final judgments.")

if not has_key():
    st.error(
        "No AI key found. Add GEMINI_API_KEY (and optionally GROQ_API_KEY) to a local .env "
        "file, or to the app's Secrets on Streamlit Cloud, then reload."
    )
    st.stop()

DEMO_TEXT = """Modern language models build on a handful of papers.
The Transformer achieved 28.4 BLEU on the WMT 2014 English-to-German translation task [1].
The Transformer relies on recurrent and convolutional layers throughout its architecture [1].
BERT obtained new state-of-the-art results on eleven natural language processing tasks [2].
BERT improved ImageNet image classification accuracy by 12 percent [2].
Quantum cabbage networks cut protein folding time by 90 percent [3].

References
[1] Vaswani, A. et al. (2017). Attention Is All You Need.
[2] Devlin, J. et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.
[3] Smith, J. (2031). Quantum Cabbage Networks for Protein Folding.
"""

REVIEW_OPTIONS = ["Not reviewed", "Accepted", "Rejected", "Needs follow-up"]

# ---------------- sidebar ----------------
max_claims = st.sidebar.slider("Max claims to check", 3, 10, MAX_CLAIMS)
st.sidebar.caption("More claims = slower run and more API usage.")
with st.sidebar.expander("Diagnostics (API errors)"):
    if ERRORS:
        for line in ERRORS[-10:]:
            st.text(line)
    else:
        st.write("No API errors so far.")


# ---------------- helpers ----------------
def badge(verdict):
    color, bg, icon = VERDICT_STYLE.get(verdict, ("#475569", "#e2e8f0", "•"))
    return (
        '<span class="cv-badge" style="color:%s;background:%s">%s %s</span>'
        % (color, bg, icon, html.escape(str(verdict)))
    )


def show_verdict(verdict, reason, confidence, model):
    color = VERDICT_STYLE.get(verdict, ("#475569", "#e2e8f0", "•"))[0]
    st.markdown(
        '<div class="cv-box" style="border-left-color:%s">%s &nbsp; '
        '<b>%s/100</b> confidence indicator &middot; model: %s<br>%s</div>'
        % (color, badge(verdict), html.escape(str(confidence)),
           html.escape(str(model)), html.escape(str(reason))),
        unsafe_allow_html=True,
    )


def log_review(i):
    """Runs when a reviewer changes the decision or note. Keeps a time-stamped history."""
    r = st.session_state["results"][i]
    decision = st.session_state.get("rev_dec_%d" % i, "Not reviewed")
    note = st.session_state.get("rev_note_%d" % i, "")
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    r["review"] = {"decision": decision, "note": note, "time": stamp}
    r["history"].append("%s | %s | %s" % (stamp, decision, note or "-"))


def load_demo():
    st.session_state["pasted_text"] = DEMO_TEXT


# ---------------- input ----------------
tab_text, tab_pdf = st.tabs(["Paste text", "Upload PDF"])
with tab_text:
    st.text_area("Paste text and its reference list", height=260, key="pasted_text")
    st.button("Load demo text", on_click=load_demo)
with tab_pdf:
    pdf_file = st.file_uploader("Upload a research paper (PDF)", type=["pdf"])
    st.caption("If a PDF is uploaded, it is used first. Remove it to use pasted text.")

if st.button("Verify citations", type="primary"):
    text = ""
    if pdf_file is not None:
        text = read_pdf(pdf_file)
    elif st.session_state.get("pasted_text", "").strip():
        text = st.session_state["pasted_text"]
    if not text.strip():
        st.warning("Please paste some text or upload a readable PDF first.")
    else:
        for key in list(st.session_state.keys()):
            if key.startswith("rev_"):
                del st.session_state[key]
        bar = st.progress(0.0, text="Reading the document and finding claims...")

        def show_progress(done, total, claim):
            bar.progress(
                min(done / max(total, 1), 1.0),
                text="Checked %d of %d claims..." % (done, total),
            )

        started = time.time()
        try:
            st.session_state["results"] = analyze(text, show_progress, max_claims)
            st.session_state["took"] = time.time() - started
        except Exception as ex:
            st.session_state["results"] = None
            st.error("Something went wrong: %s" % str(ex)[:200])
        bar.empty()

results = st.session_state.get("results")

# ---------------- results ----------------
if results is not None:
    if not results:
        st.info(
            "No cited claims were found. Try text with citations and a reference list, "
            "or check Diagnostics in the sidebar for API errors."
        )
    else:
        if st.button("Get a second opinion for all claims"):
            with st.spinner("Asking a second AI..."):
                second_opinion_all(results)
            st.rerun()

        rows = []
        for r in results:
            second = r.get("second")
            review = r.get("review") or {}
            rows.append(
                {
                    "Claim": r["claim"],
                    "Citation": r["marker"],
                    "Verdict": r["verdict"],
                    "Confidence": r["confidence"],
                    "Reference check": r.get("ref_check", ""),
                    "Reason": r["reason"],
                    "Evidence": r["evidence"],
                    "Source": (r["paper"] or {}).get("link", ""),
                    "Model": r["model"],
                    "Second opinion": second["verdict"] if second else "",
                    "Second model": second["model"] if second else "",
                    "Agreement": agreement(r),
                    "Reviewer decision": review.get("decision", "Not reviewed"),
                    "Reviewer note": review.get("note", ""),
                    "Reviewed at": review.get("time", ""),
                }
            )
        df = pd.DataFrame(rows)

        st.subheader("Summary")
        needs = int((df["Verdict"] != "Supported").sum())
        took = st.session_state.get("took")
        n_ok = len(df) - needs
        n_fake = int(df["Reference check"].astype(str).str.startswith("Not found").sum())
        cards = [
            ("Claims checked", str(len(df)), "#0f172a"),
            ("Supported", str(n_ok), "#16a34a"),
            ("Need attention", str(needs), "#dc2626" if needs else "#16a34a"),
            ("References not found", str(n_fake), "#d97706" if n_fake else "#16a34a"),
            ("Time", ("%.0fs" % took) if took else "-", "#2563eb"),
        ]
        for col, (lbl, val, color) in zip(st.columns(len(cards)), cards):
            col.markdown(
                '<div class="cv-card"><div class="lbl">%s</div>'
                '<div class="val" style="color:%s">%s</div></div>' % (lbl, color, val),
                unsafe_allow_html=True,
            )

        counts = df["Verdict"].value_counts()
        chart_df = pd.DataFrame(
            {"Verdict": [v for v in VERDICT_ORDER if v in counts.index],
             "Claims": [int(counts[v]) for v in VERDICT_ORDER if v in counts.index]}
        )
        try:
            import altair as alt
            domain = list(chart_df["Verdict"])
            rng = [VERDICT_STYLE[v][0] for v in domain]
            chart = (
                alt.Chart(chart_df)
                .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
                .encode(
                    x=alt.X("Verdict:N", sort=domain, title=None,
                            axis=alt.Axis(labelAngle=0, labelLimit=140)),
                    y=alt.Y("Claims:Q", title=None, axis=alt.Axis(tickMinStep=1)),
                    color=alt.Color("Verdict:N", scale=alt.Scale(domain=domain, range=rng),
                                    legend=None),
                    tooltip=["Verdict", "Claims"],
                )
                .properties(height=240)
            )
            st.altair_chart(chart, use_container_width=True)
        except Exception:
            st.bar_chart(chart_df.set_index("Verdict"))

        only_attention = st.checkbox("Show only claims that need attention")
        shown = df if not only_attention else df[df["Verdict"] != "Supported"]
        table = shown[["Claim", "Citation", "Verdict", "Confidence", "Reference check",
                       "Agreement", "Reviewer decision"]].copy()
        table["Verdict"] = table["Verdict"].apply(
            lambda v: "%s %s" % (VERDICT_STYLE.get(v, ("", "", "•"))[2], v)
        )
        st.dataframe(
            table,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Claim": st.column_config.TextColumn(width="large"),
                "Confidence": st.column_config.ProgressColumn(
                    "Confidence", min_value=0, max_value=100, format="%d"),
            },
        )
        st.download_button(
            "Download report (CSV)",
            df.to_csv(index=False).encode("utf-8"),
            file_name="citeverify_report.csv",
            mime="text/csv",
        )

        st.subheader("Details")
        for i, r in enumerate(results):
            if only_attention and r["verdict"] == "Supported":
                continue
            with st.expander("%s Claim %d: %s" % (VERDICT_STYLE.get(r["verdict"], ("", "", "•"))[2], i + 1, r["claim"][:90]), expanded=(i == 0)):
                st.write("**Claim:** " + r["claim"])
                st.write("**Citation:** " + (r["marker"] or "-"))
                st.write("**Reference check:** " + r.get("ref_check", "-"))
                show_verdict(r["verdict"], r["reason"], r["confidence"], r["model"])
                if r["evidence"]:
                    st.write("**Evidence from the paper:** " + r["evidence"])
                paper = r["paper"]
                if paper:
                    st.write("**Matched paper:** %s (%s)" % (paper["title"], paper["year"]))
                    if paper["link"]:
                        st.write("**Source:** " + paper["link"])
                else:
                    st.write("**Source:** not found, needs human review")

                second = r.get("second")
                if second:
                    st.write("**Second opinion:**")
                    show_verdict(
                        second["verdict"], second["reason"],
                        second["confidence"], second["model"],
                    )
                    st.write("**Agreement:** " + agreement(r))
                elif r["model"] != "-":
                    if st.button("Get a second opinion", key="so%d" % i):
                        r["second"] = second_opinion(r)
                        st.rerun()

                st.markdown("---")
                st.write("**Human review**")
                st.selectbox(
                    "Your decision", REVIEW_OPTIONS,
                    index=REVIEW_OPTIONS.index((r.get("review") or {}).get("decision", "Not reviewed")),
                    key="rev_dec_%d" % i, on_change=log_review, args=(i,),
                )
                st.text_input(
                    "Note (press Enter to save)",
                    value=(r.get("review") or {}).get("note", ""),
                    key="rev_note_%d" % i, on_change=log_review, args=(i,),
                )
                if r["history"]:
                    st.caption("Review history:")
                    for line in r["history"]:
                        st.caption(line)

        st.caption(
            "Verification uses the abstract of the matched paper only. "
            "Open the source link to check the full text."
        )
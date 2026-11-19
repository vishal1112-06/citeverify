import html
import time
from datetime import datetime

import pandas as pd
import streamlit as st

from engine import (
    analyze, read_pdf, second_opinion, second_opinion_all,
    agreement, has_key, MAX_CLAIMS, ERRORS
)

st.set_page_config(page_title="CiteVerify", page_icon="🔎", layout="wide")

STYLES = {
    "Supported": ("#15803d", "#dcfce7", "✅"),
    "Partially supported": ("#b45309", "#fef3c7", "🟡"),
    "Unsupported": ("#b91c1c", "#fee2e2", "❌"),
    "Contradictory": ("#991b1b", "#fee2e2", "⛔"),
    "Possibly outdated": ("#7c3aed", "#ede9fe", "🕒"),
    "Needs human review": ("#475569", "#e2e8f0", "👀"),
}
REVIEW_OPTIONS = ["Not reviewed", "Accepted", "Rejected", "Needs follow-up"]

st.markdown("""
<style>
.block-container{padding-top:1.5rem;max-width:1250px}
.cv-hero{background:linear-gradient(120deg,#0f172a,#1e3a8a 60%,#2563eb);
border-radius:16px;padding:26px 30px;color:white;margin-bottom:18px}
.cv-hero h1{color:white;margin:0;font-size:2.15rem}
.cv-hero p{color:#dbeafe;margin:7px 0 0}
.cv-pill{display:inline-block;background:#ffffff25;color:white;border-radius:999px;
padding:4px 11px;margin:12px 7px 0 0;font-size:.82rem}
.cv-badge{display:inline-block;border-radius:999px;padding:3px 11px;font-weight:650}
.cv-box{border-left:5px solid #94a3b8;background:#f8fafc;border-radius:8px;
padding:11px 14px;margin:8px 0;color:#0f172a}
.cv-card{border:1px solid #e2e8f0;border-radius:12px;padding:13px 15px;
background:white;color:#0f172a;margin-bottom:8px}
.cv-card .lbl{font-size:.76rem;color:#64748b;text-transform:uppercase}
.cv-card .val{font-size:1.65rem;font-weight:750}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="cv-hero">
<h1>🔎 CiteVerify</h1>
<p>Does the cited paper really support the claim? Check references and evidence transparently.</p>
<span class="cv-pill">Reference matching</span>
<span class="cv-pill">Abstract evidence checking</span>
<span class="cv-pill">AI fallback</span>
<span class="cv-pill">Human review audit trail</span>
<span class="cv-pill">CSV report</span>
</div>
""", unsafe_allow_html=True)
st.caption("Screening aid only. A search failure does not prove a reference is fabricated.")

if "results" not in st.session_state:
    st.session_state.results = None
if "took" not in st.session_state:
    st.session_state.took = None

if not has_key():
    st.error("No AI key found. Add GEMINI_API_KEY or GROQ_API_KEY to .env / Streamlit Secrets.")
    st.stop()

DEMO_TEXT = """Modern language models build on several influential papers.
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

def badge(verdict):
    fg, bg, icon = STYLES.get(verdict, ("#475569", "#e2e8f0", "•"))
    return '<span class="cv-badge" style="color:%s;background:%s">%s %s</span>' % (
        fg, bg, icon, html.escape(str(verdict))
    )

def show_verdict(verdict, reason, confidence, model):
    fg = STYLES.get(verdict, ("#475569",))[0]
    st.markdown(
        '<div class="cv-box" style="border-left-color:%s">%s &nbsp; '
        '<b>%s/100</b> AI confidence indicator · model: %s<br>%s</div>' % (
            fg, badge(verdict), html.escape(str(confidence)),
            html.escape(str(model)), html.escape(str(reason))
        ),
        unsafe_allow_html=True
    )

def log_review(i):
    item = st.session_state.results[i]
    decision = st.session_state.get("rev_dec_%d" % i, "Not reviewed")
    note = st.session_state.get("rev_note_%d" % i, "")
    old = item.get("review") or {}
    if old.get("decision") != decision or old.get("note", "") != note:
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        item["review"] = {"decision": decision, "note": note, "time": stamp}
        item.setdefault("history", []).append("%s | %s | %s" % (stamp, decision, note or "-"))

def load_demo():
    st.session_state.pasted_text = DEMO_TEXT

with st.sidebar:
    st.subheader("Run settings")
    max_claims = st.slider("Maximum claims to check", 3, 50, 10)
    st.caption("Set to 50 for the complete benchmark. This can use more API calls.")
    with st.expander("Diagnostics (recent API errors)"):
        if ERRORS:
            for error in ERRORS[-12:]:
                st.text(error)
        else:
            st.write("No API errors recorded in this session.")

tab_text, tab_pdf = st.tabs(["Paste text", "Upload PDF"])
with tab_text:
    st.text_area("Paste the paper text and its reference list", height=260, key="pasted_text")
    st.button("Load demo text", on_click=load_demo)
with tab_pdf:
    pdf_file = st.file_uploader("Upload a research paper (PDF)", type=["pdf"])
    st.caption("An uploaded PDF takes priority over pasted text.")

if st.button("Verify citations", type="primary"):
    try:
        text = read_pdf(pdf_file) if pdf_file is not None else st.session_state.get("pasted_text", "")
        if not text.strip():
            st.warning("Paste text or upload a readable PDF first.")
        else:
            for key in list(st.session_state.keys()):
                if key.startswith("rev_"):
                    del st.session_state[key]
            progress_bar = st.progress(0.0, text="Extracting cited claims...")
            started = time.time()
            def progress(done, total, claim):
                progress_bar.progress(done / max(total, 1),
                                      text="Checked %d of %d claims..." % (done, total))
            try:
                st.session_state.results = analyze(text, progress, max_claims)
                st.session_state.took = time.time() - started
            finally:
                progress_bar.empty()
    except Exception as ex:
        st.session_state.results = None
        st.error("Analysis failed: %s" % str(ex)[:250])

results = st.session_state.results
if results is not None:
    if not results:
        st.error("No cited claims were extracted.")
        st.caption("Possible causes: the PDF has no extractable text, its citation style is unsupported, or AI extraction failed.")
        with st.expander("Debug extracted text"):
            st.write("Extracted characters:", len(text) if "text" in locals() else "Unavailable")
            if "text" in locals() and text:
                preview = text[:5000]
                st.text(preview + ("\\n\\n[Preview truncated]" if len(text) > 5000 else ""))
            st.write("Recent API diagnostics:")
            if ERRORS:
                for error in ERRORS[-8:]:
                    st.code(error)
            else:
                st.write("No API errors were recorded.")
    else:
        if st.button("Get second opinions for all eligible claims"):
            with st.spinner("Requesting second opinions..."):
                second_opinion_all(results)
            st.rerun()

        rows = []
        for item in results:
            paper = item.get("paper") or {}
            second = item.get("second") or {}
            review = item.get("review") or {}
            rows.append({
                "Claim": item.get("claim", ""),
                "Citation": item.get("marker", ""),
                "Verdict": item.get("verdict", ""),
                "Confidence": item.get("confidence", 0),
                "Reference check": item.get("ref_check", ""),
                "Title match": item.get("title_match", ""),
                "Year check": item.get("year_check", ""),
                "Reason": item.get("reason", ""),
                "Evidence": item.get("evidence", ""),
                "Source": paper.get("link", ""),
                "Matched title": paper.get("title", ""),
                "Matched year": paper.get("year", ""),
                "Search provider": paper.get("source", ""),
                "Model": item.get("model", "-"),
                "Second opinion": second.get("verdict", ""),
                "Second model": second.get("model", ""),
                "Agreement": agreement(item),
                "Reviewer decision": review.get("decision", "Not reviewed"),
                "Reviewer note": review.get("note", ""),
                "Reviewed at": review.get("time", ""),
            })
        df = pd.DataFrame(rows)

        st.subheader("Summary")
        total = len(df)
        supported = int((df["Verdict"] == "Supported").sum())
        attention = total - supported
        unresolved = int((df["Verdict"] == "Needs human review").sum())
        not_found = int(df["Reference check"].astype(str).str.startswith("No reliable").sum())
        cards = [
            ("Claims checked", total, "#0f172a"),
            ("Supported", supported, "#15803d"),
            ("Need attention", attention, "#b91c1c"),
            ("Human review", unresolved, "#475569"),
            ("No reliable match", not_found, "#b45309"),
            ("Run time", "%.1fs" % st.session_state.took if st.session_state.took is not None else "-", "#1d4ed8"),
        ]
        for col, (label, value, color) in zip(st.columns(len(cards)), cards):
            col.markdown('<div class="cv-card"><div class="lbl">%s</div><div class="val" style="color:%s">%s</div></div>' %
                         (html.escape(label), color, html.escape(str(value))), unsafe_allow_html=True)

        counts = df["Verdict"].value_counts()
        chart_df = pd.DataFrame({
            "Verdict": [v for v in STYLES if v in counts.index],
            "Claims": [int(counts[v]) for v in STYLES if v in counts.index]
        })
        if not chart_df.empty:
            try:
                import altair as alt
                domain = chart_df["Verdict"].tolist()
                chart = alt.Chart(chart_df).mark_bar().encode(
                    x=alt.X("Verdict:N", sort=domain, title=None),
                    y=alt.Y("Claims:Q", title=None),
                    tooltip=["Verdict", "Claims"]
                ).properties(height=230)
                st.altair_chart(chart, use_container_width=True)
            except Exception:
                st.bar_chart(chart_df.set_index("Verdict"))

        only_attention = st.checkbox("Show only claims that need attention")
        shown = df if not only_attention else df[df["Verdict"] != "Supported"]
        st.dataframe(shown[[
            "Claim", "Citation", "Verdict", "Confidence", "Reference check",
            "Title match", "Year check", "Agreement", "Reviewer decision"
        ]], hide_index=True, use_container_width=True,
        column_config={"Claim": st.column_config.TextColumn(width="large"),
                       "Confidence": st.column_config.ProgressColumn(
                           "AI confidence indicator", min_value=0, max_value=100, format="%d")})

        st.download_button(
            "Download detailed report (CSV)",
            df.to_csv(index=False).encode("utf-8-sig"),
            file_name="citeverify_report.csv",
            mime="text/csv"
        )

        st.subheader("Claim details")
        for i, item in enumerate(results):
            if only_attention and item["verdict"] == "Supported":
                continue
            icon = STYLES.get(item["verdict"], ("", "", "•"))[2]
            with st.expander("%s Claim %d: %s" % (icon, i + 1, item.get("claim", "")[:100]),
                             expanded=(i == 0)):
                st.write("**Claim:** " + item.get("claim", ""))
                st.write("**Citation:** " + (item.get("marker") or "-"))
                st.write("**Reference check:** " + item.get("ref_check", "-"))
                st.write("**Title match:** " + str(item.get("title_match", "-")))
                st.write("**Year check:** " + str(item.get("year_check", "-")))
                show_verdict(item["verdict"], item["reason"], item["confidence"], item["model"])
                if item.get("evidence"):
                    st.write("**Exact evidence from abstract:** " + item["evidence"])
                paper = item.get("paper")
                if paper:
                    st.write("**Matched paper:** %s (%s)" % (paper.get("title"), paper.get("year") or "year unknown"))
                    st.write("**Search provider:** " + paper.get("source", ""))
                    if paper.get("link"):
                        st.markdown("[Open source record](%s)" % paper["link"])
                else:
                    st.write("**Source:** no reliable match; human review required")

                second = item.get("second")
                if second:
                    st.write("**Second opinion**")
                    show_verdict(second["verdict"], second["reason"], second["confidence"], second["model"])
                    st.write("**Agreement:** " + agreement(item))
                elif item.get("model") != "-":
                    if st.button("Get second opinion", key="so_%d" % i):
                        item["second"] = second_opinion(item)
                        st.rerun()

                st.divider()
                st.write("**Human review**")
                review = item.get("review") or {}
                old_decision = review.get("decision", "Not reviewed")
                st.selectbox("Reviewer decision", REVIEW_OPTIONS,
                             index=REVIEW_OPTIONS.index(old_decision) if old_decision in REVIEW_OPTIONS else 0,
                             key="rev_dec_%d" % i, on_change=log_review, args=(i,))
                st.text_input("Reviewer note (press Enter to save)",
                              value=review.get("note", ""),
                              key="rev_note_%d" % i, on_change=log_review, args=(i,))
                for line in item.get("history", []):
                    st.caption(line)

        st.caption(
            "Limitation: claim support is assessed against the matched paper's abstract only, "
            "not its complete text. Confidence is an AI indicator, not a calibrated probability."
        )

import time
from datetime import datetime

import pandas as pd
import streamlit as st

from engine import (
    analyze, read_pdf, second_opinion, second_opinion_all, agreement,
    has_key, MAX_CLAIMS, ERRORS,
)

st.set_page_config(page_title="CiteVerify", layout="wide")

st.title("CiteVerify")
st.caption(
    "Checks whether cited papers really support the claims made in a document. "
    "Results are indicators for human review, not final judgments."
)

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
def show_verdict(verdict, reason, confidence, model):
    message = "**%s** (confidence indicator: %s/100, model: %s). %s" % (
        verdict, confidence, model, reason,
    )
    if verdict == "Supported":
        st.success(message)
    elif verdict in ("Unsupported", "Contradictory"):
        st.error(message)
    else:
        st.warning(message)


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
        st.write(
            "**%d of %d claims need attention or human review.**%s"
            % (needs, len(df), (" Checked in %.0f seconds." % took) if took else "")
        )
        st.bar_chart(df["Verdict"].value_counts())
        only_attention = st.checkbox("Show only claims that need attention")
        shown = df if not only_attention else df[df["Verdict"] != "Supported"]
        st.dataframe(
            shown[["Claim", "Citation", "Verdict", "Confidence", "Reference check",
                   "Agreement", "Reviewer decision"]],
            hide_index=True,
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
            with st.expander("Claim %d: %s" % (i + 1, r["claim"][:90]), expanded=(i == 0)):
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
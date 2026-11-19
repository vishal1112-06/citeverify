from datetime import datetime

import pandas as pd
import streamlit as st

from engine import analyze, read_pdf, second_opinion, agreement, has_key, MAX_CLAIMS

st.set_page_config(page_title="CiteVerify", layout="wide")

st.title("CiteVerify")
st.caption(
    "Checks whether cited papers really support the claims made in a document. "
    "Results are indicators for human review, not final judgments."
)

if not has_key():
    st.error(
        "GEMINI_API_KEY is missing. Add it to a local .env file, or to the app's "
        "Secrets on Streamlit Cloud, then reload."
    )
    st.stop()

max_claims = st.sidebar.slider("Max claims to check", 3, 10, MAX_CLAIMS)
st.sidebar.caption("More claims = slower run and more API usage.")

REVIEW_OPTIONS = ["Not reviewed", "Accepted", "Rejected", "Needs follow-up"]


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
    """Called when a reviewer changes the decision or note. Keeps a time-stamped history."""
    r = st.session_state["results"][i]
    decision = st.session_state.get("rev_dec_%d" % i, "Not reviewed")
    note = st.session_state.get("rev_note_%d" % i, "")
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    r["review"] = {"decision": decision, "note": note, "time": stamp}
    r["history"].append("%s | %s | %s" % (stamp, decision, note or "-"))


tab_text, tab_pdf = st.tabs(["Paste text", "Upload PDF"])
with tab_text:
    pasted = st.text_area("Paste text and its reference list", height=260)
with tab_pdf:
    pdf_file = st.file_uploader("Upload a research paper (PDF)", type=["pdf"])
    st.caption("If a PDF is uploaded, it is used first. Remove it to use pasted text.")

if st.button("Verify citations", type="primary"):
    text = ""
    if pdf_file is not None:
        text = read_pdf(pdf_file)
    elif pasted.strip():
        text = pasted
    if not text.strip():
        st.warning("Please paste some text or upload a readable PDF first.")
    else:
        for key in list(st.session_state.keys()):
            if key.startswith("rev_"):
                del st.session_state[key]
        bar = st.progress(0.0, text="Starting...")

        def show_progress(i, total, claim):
            bar.progress(
                i / max(total, 1),
                text="Checking claim %d of %d..." % (i + 1, total),
            )

        st.session_state["results"] = analyze(text, show_progress, max_claims)
        bar.empty()

results = st.session_state.get("results")

if results is not None:
    if not results:
        st.info("No cited claims were found. Try text with citations and a reference list.")
    else:
        if st.button("Get a second opinion for all claims"):
            with st.spinner("Asking the second model..."):
                for r in results:
                    if not r.get("second") and r["model"] != "-":
                        r["second"] = second_opinion(r)
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
                    "Agreement": agreement(r),
                    "Reviewer decision": review.get("decision", "Not reviewed"),
                    "Reviewer note": review.get("note", ""),
                    "Reviewed at": review.get("time", ""),
                }
            )
        df = pd.DataFrame(rows)

        st.subheader("Summary")
        counts = df["Verdict"].value_counts()
        st.write(
            "  |  ".join("%s: %d" % (k, v) for k, v in counts.items())
        )
        needs = int((df["Verdict"] != "Supported").sum())
        st.caption("%d of %d claims need attention or human review." % (needs, len(df)))
        st.dataframe(
            df[["Claim", "Citation", "Verdict", "Confidence", "Reference check", "Agreement", "Reviewer decision"]]
        )
        st.download_button(
            "Download report (CSV)",
            df.to_csv(index=False).encode("utf-8"),
            file_name="citeverify_report.csv",
            mime="text/csv",
        )

        st.subheader("Details")
        for i, r in enumerate(results):
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

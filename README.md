# CiteVerify - Scientific Claim and Citation Verification System

Innovedge Hackathon 2026, Problem Statement 03.

Upload a PDF or paste text with a reference list. CiteVerify extracts cited claims, finds the
cited papers (OpenAlex, Semantic Scholar), compares each claim with the paper's abstract using
Gemini, and labels it Supported / Partially supported / Unsupported / Contradictory /
Possibly outdated / Needs human review, with reason, evidence quote, confidence indicator and
source link. A second, different Gemini model can re-check any claim; disagreements are flagged
for human review. Reviewers can record a decision and note, kept as a time-stamped history.

Results are indicators for human review, not final academic judgments.

## Run locally
1. `pip install -r requirements.txt`
2. Create a `.env` file containing `GEMINI_API_KEY=your_key`
3. `streamlit run app.py`
4. Paste `sample_demo.txt` and click Verify citations.

## Deploy (Streamlit Community Cloud)
Main file `app.py`; add `GEMINI_API_KEY = "your_key"` under Settings > Secrets.

## Known limits
- Verification uses the abstract only, not the full text.
- Both checkers are Gemini models, so they are not fully independent.
- Free API tiers can be rate limited; the engine retries and falls back to a lighter model.

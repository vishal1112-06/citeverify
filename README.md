# CiteVerify

**Does the cited paper really say that?** CiteVerify checks every cited claim in a document against the paper it cites. It flags references that may be fake or wrong, shows the exact evidence behind each verdict, and leaves the final decision to a human reviewer.

- **Live app:** https://citeverify.streamlit.app
- **Code:** https://github.com/vishal1112-06/citeverify
- **Hackathon:** Innovedge Hackathon 2026, Problem Statement 03
- **Author:** Vishy, B.E. CSE (AI and ML), Sapthagiri NPS University

## The problem

Research writing is full of claims like "Method X achieved 28.4 BLEU [1]". Readers rarely open reference [1]. Citations get exaggerated, misquoted, or invented, and AI-written text makes this worse because it can cite papers that do not exist. Checking by hand is slow, so most citation errors survive review.

## The solution

Paste text with its reference list, or upload a PDF. For every cited claim, CiteVerify:

1. **Finds the paper.** It looks up the reference on OpenAlex, then arXiv, Crossref and Semantic Scholar. If no service can find it, the reference is flagged "Not found online - reference may be wrong".
2. **Compares claim and evidence.** An AI model reads the paper's abstract and returns one verdict: Supported, Partially supported, Unsupported, Contradictory, or Needs human review.
3. **Proves its evidence.** The model must quote the abstract. If the quote does not appear in the abstract, it is removed and the confidence is capped. This blocks made-up evidence.
4. **Offers a second opinion.** A second model re-checks the claim, and the app shows whether the two agree. When both used the same model, it says so.
5. **Keeps a human in charge.** The reviewer accepts, rejects or flags each claim and adds a note. Every decision is time-stamped. The full report downloads as CSV.

## Why it is different

| Typical "AI fact checker" | CiteVerify |
|---|---|
| One model, trusts its own answer | Quote guard: evidence must exist in the source text |
| Answers even when unsure | Says "Needs human review" or "Could not check" instead of guessing |
| Does not check the reference exists | Looks the paper up on four services and flags missing references |
| Verdict only | Verdict, evidence quote, source link, confidence, second opinion |
| No audit trail | Time-stamped human decisions and a CSV report |

## Demo (30 seconds)

Open the live app, click **Load demo text**, set "Max claims to check" to 5 or more, then click **Verify citations**.

| Claim in the demo text | Result |
|---|---|
| Transformer achieved 28.4 BLEU on WMT 2014 En-De | Supported |
| Transformer relies on recurrent and convolutional layers | Contradictory (the abstract says it dispenses with them) |
| BERT set state of the art on eleven NLP tasks | Supported |
| BERT improved ImageNet accuracy by 12 percent | Unsupported |
| Quantum cabbage networks cut protein folding time by 90 percent | Reference not found, needs human review |

All five verdicts were correct in our test runs. This is a small hand-built sample, not a benchmark. See the limits below.

## How it works

```
text or PDF
  -> find cited claims and match them to the reference list
  -> look up each paper (OpenAlex, arXiv, Crossref, Semantic Scholar)
  -> AI verdict from the abstract
  -> quote guard and confidence caps
  -> optional second model
  -> human review log
  -> CSV report
```

- `app.py`: Streamlit interface (summary cards, chart, table, claim details, review form).
- `engine.py`: the pipeline (claim extraction, paper lookup, verification, quote guard, second opinion).
- Claims run in parallel. If a model hits a rate limit, the app marks it unavailable for 2 minutes and tries another. A Diagnostics panel in the sidebar shows API errors.

**Confidence rules.** No evidence quote means a maximum of 60. A "Partially supported" verdict means a maximum of 80. A quote that is not found in the abstract is removed.

## Responsible AI

- Results are indicators to support review, not final judgments, and the app says so.
- The app never invents a source. Failed lookups become "Needs human review" or "Could not check".
- Every claim shows a source link so a person can read the paper.
- No login, no database, and no document is stored by the app.

## Honest limits

- **Abstract only.** Verdicts use the paper's abstract, not its full text. A true claim that appears only in the body can come back Unsupported or Needs human review.
- **Small test set.** We tested a short demo sample and a few PDFs, not a large benchmark. Accuracy on other documents is not yet measured.
- **Free-tier services.** The AI and paper-search APIs are free tiers, so rate limits can slow a run (about 40 seconds to 2 minutes for 5 claims) or force a fallback model.
- **Same-vendor models.** Two Gemini models count as one source of opinion, and the app labels them that way. A second vendor (Groq) is used when its key is configured.
- **Confidence is a guide,** not a probability.

## Roadmap

1. Full-text checking, with the matching passage highlighted.
2. Highlight claims inside the uploaded PDF.
3. Batch processing of long documents, with a "check next batch" button.
4. A labelled benchmark to measure precision and recall.

## Run it locally

```
pip install -r requirements.txt
```

Create a `.env` file in the project folder (no quotes around values):

```
GEMINI_API_KEY=your_key
OPENALEX_API_KEY=your_key
GROQ_API_KEY=your_key
```

`GROQ_API_KEY` is optional. Then run:

```
python -m streamlit run app.py
```

On Streamlit Community Cloud, add the same names under the app's Secrets (TOML format, values in quotes). Never commit `.env`; it is in `.gitignore`.

## Tech

Streamlit, Google Gemini, Groq, OpenAlex, arXiv, Crossref, Semantic Scholar, pypdf, pandas, requests.
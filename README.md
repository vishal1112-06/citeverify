# CiteVerify

**Does the cited paper really say that?** CiteVerify checks every cited claim in a document against the paper it cites, flags references that may be fake or wrong, and leaves the final call to a human reviewer.

- **Live app:** https://citeverify.streamlit.app
- **Code:** https://github.com/vishal1112-06/citeverify
- **Built for:** Innovedge Hackathon 2026, Problem Statement 03

## The problem

Scientific writing is full of claims like "Method X achieved 28.4 BLEU [1]". Readers rarely open reference [1] to check. Citations get misquoted, exaggerated, or invented, and AI-written text makes this worse because it can produce references that do not exist. Checking by hand is slow.

## What CiteVerify does

1. **Finds the cited claims.** Paste text with its reference list, or upload a PDF.
2. **Checks the reference exists.** It looks the paper up on OpenAlex, then arXiv, Crossref and Semantic Scholar. A reference that none of them can find is flagged "Not found online - reference may be wrong".
3. **Compares claim with evidence.** An AI model reads the paper's abstract and returns one verdict: Supported, Partially supported, Unsupported, Contradictory, or Needs human review.
4. **Guards against made-up evidence.** The model must quote the abstract. If its quote does not appear in the abstract, the quote is removed and the confidence is capped.
5. **Offers a second opinion.** A second model, from a different vendor when a Groq key is configured, re-checks the claim. The app shows whether the two agree, and says plainly when both used the same model.
6. **Keeps a human in charge.** The reviewer accepts, rejects or flags each claim, adds a note, and every decision is time-stamped. The full report downloads as CSV.

## Demo (30 seconds)

Click **Load demo text**, then **Verify citations**. Use at least 5 for "Max claims to check" in the sidebar. Expected results:

| Claim | Expected result |
|---|---|
| Transformer achieved 28.4 BLEU on WMT 2014 En-De | Supported |
| Transformer relies on recurrent and convolutional layers | Contradictory |
| BERT set new state of the art on eleven NLP tasks | Supported |
| BERT improved ImageNet accuracy by 12 percent | Unsupported |
| Quantum cabbage networks cut protein folding time by 90 percent | Needs human review, reference not found |

## How it works

```
text or PDF -> find cited claims -> look up each paper (OpenAlex, arXiv, Crossref, Semantic Scholar)
            -> AI verdict from the abstract -> quote check -> confidence caps
            -> optional second model -> human review -> CSV report
```

- `app.py` is the Streamlit interface.
- `engine.py` holds the pipeline: claim extraction, paper lookup, verification, the quote guard and the second opinion.
- Claims are checked in parallel. If a model hits a rate limit, the app marks it unavailable for 2 minutes and tries another one.

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

`GROQ_API_KEY` is optional and enables a second model from a different vendor. Then run:

```
python -m streamlit run app.py
```

On Streamlit Community Cloud, put the same names in the app's Secrets (TOML, with quotes). Never commit `.env`; it is listed in `.gitignore`.

## Honest limits

- **Abstract only.** Verdicts are based on the paper's abstract, not its full text. A claim that is true but only stated in the body can come back as Unsupported or Needs human review. This is why the app shows a source link for every match.
- **Indicators, not judgments.** The confidence number is a guide for the reviewer. It is not a probability.
- **Free-tier limits.** The AI and paper-lookup services are free tiers, so a busy moment can cause rate limits and slow runs (about 40 seconds to 2 minutes for 5 claims). When a service fails, the app marks the claim "Needs human review" or "Could not check". It never guesses.
- **Not tested at scale.** Tested on short samples and a few PDFs, not on large document sets.
- **Models.** Gemini Flash and Flash-Lite are the same vendor, so they count as one source of opinion, and the app labels them that way.

## Tech

Streamlit, Google Gemini, Groq, OpenAlex, arXiv, Crossref, Semantic Scholar, pypdf, pandas, requests.
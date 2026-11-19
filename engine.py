"""CiteVerify engine: extract claims, find cited papers, verify, second opinion."""
import os
import re
import json
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from difflib import SequenceMatcher

import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pypdf import PdfReader

load_dotenv()

MODEL_A = "gemini-3.5-flash"
MODEL_B = "gemini-3.5-flash-lite"
MAX_CLAIMS = 6

VERDICTS = [
    "Supported",
    "Partially supported",
    "Unsupported",
    "Contradictory",
    "Possibly outdated",
    "Needs human review",
]


def get_secret(name):
    value = os.getenv(name)
    if value:
        return value.strip()
    try:
        import streamlit as st
        return str(st.secrets[name]).strip()
    except Exception:
        return None


def get_key():
    return get_secret("GEMINI_API_KEY")


def has_key():
    return bool(get_key())


_client = None


def get_client():
    """Create the Gemini client on first use, so a missing key shows a clear message
    in the app instead of crashing the whole page at import time."""
    global _client
    if _client is None:
        _client = genai.Client(api_key=get_key())
    return _client


def ask_gemini(prompt, models, as_json=False):
    """Try each model in order. Returns (answer_text, model_used) or (None, None)."""
    config = None
    if as_json:
        config = types.GenerateContentConfig(response_mime_type="application/json")
    try:
        client = get_client()
    except Exception:
        return None, None
    for model in models:
        for attempt in range(2):
            try:
                r = client.models.generate_content(
                    model=model, contents=prompt, config=config
                )
                return r.text, model
            except Exception:
                time.sleep(2)
    return None, None


def parse_json(text):
    if not text:
        return None
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except Exception:
        return None


def read_pdf(file):
    reader = PdfReader(file)
    pages = [(page.extract_text() or "") for page in reader.pages]
    return "\n".join(pages)


def split_text(text, max_body=20000, max_refs=8000):
    """Split a document into body and reference list (last 'References' heading)."""
    pattern = r"\n\s*(references|bibliography)\s*\n"
    matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))
    if matches:
        idx = matches[-1].start()
        body, refs = text[:idx], text[idx:]
    else:
        body, refs = text, ""
    return body[:max_body], refs[:max_refs]


def extract_claims(text, max_claims=MAX_CLAIMS):
    body, refs = split_text(text)
    prompt = (
        "You are a careful research assistant.\n"
        "From the DOCUMENT BODY, pick up to " + str(max_claims) + " important factual\n"
        "claims (statements, numbers or conclusions) that cite a source.\n"
        "For each claim, find the matching entry in the REFERENCE LIST.\n"
        "Return ONLY a JSON list. Each item must have these keys:\n"
        "claim: the exact sentence from the document\n"
        "marker: the citation marker, such as [1] or (Smith, 2020)\n"
        "reference: the full matching reference text, or an empty string if not found\n"
        "title: the cited paper title if you can read it, or an empty string\n"
        "Do not invent references.\n\n"
        "DOCUMENT BODY:\n" + body + "\n\n"
        "REFERENCE LIST:\n" + (refs or "(none found)")
    )
    answer, model = ask_gemini(prompt, [MODEL_A, MODEL_B], as_json=True)
    data = parse_json(answer)
    if isinstance(data, dict):
        for value in data.values():
            if isinstance(value, list):
                data = value
                break
    if not isinstance(data, list):
        return []
    claims = []
    for item in data:
        if isinstance(item, dict) and item.get("claim"):
            claims.append(
                {
                    "claim": str(item.get("claim", "")).strip(),
                    "marker": str(item.get("marker", "")).strip(),
                    "reference": str(item.get("reference", "")).strip(),
                    "title": str(item.get("title", "")).strip(),
                }
            )
    return claims[:max_claims]


def looks_same(found, query):
    found = (found or "").lower().strip()
    query = (query or "").lower().strip()
    if not found or not query:
        return False
    if found in query:
        return True
    return SequenceMatcher(None, found, query).ratio() > 0.6


def rebuild_abstract(inverted):
    """OpenAlex stores abstracts as {word: [positions]}. Rebuild the sentence."""
    if not inverted:
        return ""
    words = {}
    for word, positions in inverted.items():
        for p in positions:
            words[p] = word
    return " ".join(words[i] for i in sorted(words))


def http_get(url, params=None, headers=None):
    """GET with one polite retry when the service says 429 (too many requests)."""
    r = requests.get(url, params=params, headers=headers, timeout=10)
    if r.status_code == 429:
        time.sleep(3)
        r = requests.get(url, params=params, headers=headers, timeout=10)
    return r


def search_openalex(query):
    params = {"search": query, "per_page": 1}
    key = get_secret("OPENALEX_API_KEY")
    if key:
        params["api_key"] = key
    r = http_get("https://api.openalex.org/works", params=params)
    r.raise_for_status()
    results = r.json().get("results", [])
    if not results:
        return None
    w = results[0]
    return {
        "title": w.get("title") or "",
        "year": w.get("publication_year"),
        "abstract": rebuild_abstract(w.get("abstract_inverted_index")),
        "link": w.get("doi") or w.get("id") or "",
        "source": "OpenAlex",
    }


def search_semantic_scholar(query):
    key = get_secret("SEMANTIC_SCHOLAR_API_KEY")
    r = http_get(
        "https://api.semanticscholar.org/graph/v1/paper/search",
        params={"query": query, "limit": 1, "fields": "title,abstract,year,url"},
        headers={"x-api-key": key} if key else None,
    )
    r.raise_for_status()
    data = r.json().get("data", [])
    if not data:
        return None
    p = data[0]
    return {
        "title": p.get("title") or "",
        "year": p.get("year"),
        "abstract": p.get("abstract") or "",
        "link": p.get("url") or "",
        "source": "Semantic Scholar",
    }


def search_arxiv(query):
    """arXiv needs no key and gives abstracts for papers posted there (most CS/AI papers)."""
    q = re.sub(r"[^A-Za-z0-9 ]+", " ", query)[:200].strip()
    r = requests.get(
        "https://export.arxiv.org/api/query",
        params={"search_query": 'all:"%s"' % q, "max_results": 1},
        timeout=15,
    )
    r.raise_for_status()
    return parse_arxiv(r.text)


def parse_arxiv(xml_text):
    ns = {"a": "http://www.w3.org/2005/Atom"}
    entry = ET.fromstring(xml_text).find("a:entry", ns)
    if entry is None:
        return None
    clean = lambda t: re.sub(r"\s+", " ", t or "").strip()
    published = clean(entry.findtext("a:published", default="", namespaces=ns))
    return {
        "title": clean(entry.findtext("a:title", default="", namespaces=ns)),
        "year": int(published[:4]) if published[:4].isdigit() else None,
        "abstract": clean(entry.findtext("a:summary", default="", namespaces=ns)),
        "link": clean(entry.findtext("a:id", default="", namespaces=ns)),
        "source": "arXiv",
    }


def search_crossref(query):
    r = requests.get(
        "https://api.crossref.org/works",
        params={"query.bibliographic": query[:300], "rows": 1},
        timeout=10,
    )
    r.raise_for_status()
    items = r.json().get("message", {}).get("items", [])
    if not items:
        return None
    w = items[0]
    year = None
    for k in ("issued", "published-print", "published-online"):
        parts = (w.get(k) or {}).get("date-parts") or [[None]]
        if parts and parts[0] and parts[0][0]:
            year = parts[0][0]
            break
    abstract = re.sub(r"<[^>]+>", " ", w.get("abstract") or "")
    return {
        "title": ((w.get("title") or [""])[0]),
        "year": year,
        "abstract": re.sub(r"\s+", " ", abstract).strip(),
        "link": ("https://doi.org/" + w["DOI"]) if w.get("DOI") else "",
        "source": "Crossref",
    }


_paper_cache = {}


def lookup_paper(title, reference, notes=None):
    """Find the cited paper online. Returns a dict or None.
    Same paper is looked up only once. Why a search failed is appended to notes."""
    query = title or (reference or "")[:300]
    if not query:
        return None
    key = (title or "").lower() + "|" + (reference or "")[:120].lower()
    if key in _paper_cache:
        return _paper_cache[key]
    check_against = (title or "") + " " + (reference or "")
    best = None
    found = None
    searchers = (
        ("OpenAlex", search_openalex),
        ("arXiv", search_arxiv),
        ("Crossref", search_crossref),
        ("Semantic Scholar", search_semantic_scholar),
    )
    for name, search in searchers:
        try:
            paper = search(query)
            if paper is None and notes is not None:
                notes.append(name + ": no result")
        except Exception as ex:
            paper = None
            if notes is not None:
                notes.append(name + ": " + str(ex)[:60])
        if paper and looks_same(paper["title"], check_against):
            if paper["abstract"]:
                found = paper
                break
            if best is None:
                best = paper
        elif paper and notes is not None:
            notes.append(name + ": top hit did not match (" + paper["title"][:40] + ")")
        time.sleep(0.3)
    result = found or best
    if result is not None:
        _paper_cache[key] = result
    return result


def _norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[\"'“”‘’]", "", s or "")).strip().lower()


def quote_in_abstract(quote, abstract):
    """True only if the quoted evidence really appears in the abstract.
    Guards against the AI inventing a quote. '...' inside a quote is allowed."""
    parts = [p for p in re.split(r"\.\.\.|…", quote or "") if p.strip()]
    if not parts:
        return False
    a = _norm(abstract)
    return all(_norm(p).strip(" .") in a for p in parts)


def verify_claim(claim, paper, models):
    """Compare a claim with the abstract of its cited paper."""
    if not paper or not paper.get("abstract"):
        return {
            "verdict": "Needs human review",
            "reason": "The cited paper or its abstract could not be found automatically.",
            "evidence": "",
            "confidence": 0,
            "model": "-",
        }
    prompt = (
        "You are checking whether a cited paper supports a claim.\n"
        "Use ONLY the abstract below. Do not use outside knowledge.\n"
        "Choose one verdict: Supported, Partially supported, Unsupported,\n"
        "Contradictory, Possibly outdated, Needs human review.\n"
        "Use Possibly outdated only if the claim is about the current state of a fast-moving\n"
        "field and the paper is old (today's year is " + str(datetime.now().year) + ").\n"
        "Return ONLY a JSON object with these keys:\n"
        "verdict: one of the verdicts above\n"
        "reason: one or two short sentences explaining why\n"
        "evidence: a short quote copied word for word from the abstract, or an empty string\n"
        "confidence: an integer from 0 to 100 showing how strong the evidence is\n"
        "If the abstract does not contain enough information, use Needs human review.\n\n"
        "CLAIM: " + claim + "\n\n"
        "PAPER TITLE: " + paper["title"] + "\n"
        "PAPER YEAR: " + str(paper.get("year")) + "\n"
        "ABSTRACT: " + paper["abstract"]
    )
    answer, used = ask_gemini(prompt, models, as_json=True)
    data = parse_json(answer)
    if isinstance(data, list) and data:
        data = data[0]
    if not isinstance(data, dict):
        return {
            "verdict": "Needs human review",
            "reason": "The AI model did not return a usable answer.",
            "evidence": "",
            "confidence": 0,
            "model": used or "-",
        }
    verdict = data.get("verdict")
    if verdict not in VERDICTS:
        verdict = "Needs human review"
    try:
        confidence = max(0, min(100, int(data.get("confidence", 0))))
    except Exception:
        confidence = 0
    reason = str(data.get("reason", "")).strip()
    evidence = str(data.get("evidence", "")).strip()
    if evidence and not quote_in_abstract(evidence, paper["abstract"]):
        evidence = ""
        confidence = min(confidence, 50)
        reason += " (The quoted evidence did not match the abstract, so it was removed.)"
    return {
        "verdict": verdict,
        "reason": reason,
        "evidence": evidence,
        "confidence": confidence,
        "model": used or "-",
    }


def reference_check(c, paper, notes=None):
    """Is the cited paper real and findable? Flags missing or possibly wrong references."""
    if paper:
        return "Found on " + paper.get("source", "the web")
    if not c.get("reference") and not c.get("title"):
        return "No reference listed for this claim"
    notes = notes or []
    errors = [n for n in notes if "no result" not in n and "did not match" not in n]
    why = (" [" + "; ".join(notes) + "]") if notes else ""
    if errors:
        return "Could not check - paper search service error, retry later" + why
    return "Not found online - reference may be wrong" + why


def analyze(text, progress=None, max_claims=MAX_CLAIMS):
    """Full pipeline: claims, paper lookup, verdicts. Returns a list of dicts."""
    claims = extract_claims(text, max_claims)
    results = []
    for i, c in enumerate(claims):
        if progress:
            progress(i, len(claims), c["claim"])
        notes = []
        paper = lookup_paper(c["title"], c["reference"], notes)
        verdict = verify_claim(c["claim"], paper, [MODEL_A, MODEL_B])
        item = dict(c)
        item.update(verdict)
        item["paper"] = paper
        item["ref_check"] = reference_check(c, paper, notes)
        item["second"] = None
        item["review"] = None
        item["history"] = []
        results.append(item)
    return results


def second_opinion(item):
    """Re-check one claim with the OTHER model, so the two opinions are independent."""
    other = MODEL_B if item.get("model") == MODEL_A else MODEL_A
    return verify_claim(item["claim"], item.get("paper"), [other])


def agreement(item):
    second = item.get("second")
    if not second:
        return ""
    if second["model"] in ("-", None):
        return "No second opinion available"
    if second["verdict"] == item["verdict"]:
        return "Agree"
    return "Disagree - needs human review"

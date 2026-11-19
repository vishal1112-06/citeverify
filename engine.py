"""CiteVerify: citation extraction, strict metadata matching, abstract checking and AI fallback."""
import json
import os
import re
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher

import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pypdf import PdfReader

load_dotenv()

MODEL_A = os.getenv("GEMINI_MODEL_PRIMARY", "gemini-2.5-flash")
MODEL_B = os.getenv("GEMINI_MODEL_FALLBACK", "gemini-2.5-flash-lite")
MODEL_GROQ = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
MAX_CLAIMS = 50
TITLE_MATCH_THRESHOLD = 0.84
YEAR_TOLERANCE = 2
VERDICTS = {"Supported", "Partially supported", "Unsupported", "Contradictory",
            "Possibly outdated", "Needs human review"}

ERRORS = []
_clients = None
_dead = {}
_paper_cache = {}


def _log_error(message):
    ERRORS.append(str(message)[:240])
    del ERRORS[:-30]


def get_secret(name):
    value = os.getenv(name)
    if value:
        return value.strip()
    try:
        import streamlit as st
        value = st.secrets.get(name)
        return str(value).strip() if value else None
    except Exception:
        return None


def get_keys():
    return [v for v in (get_secret("GEMINI_API_KEY"), get_secret("GEMINI_API_KEY_2")) if v]


def has_key():
    return bool(get_keys() or get_secret("GROQ_API_KEY"))


def get_clients():
    global _clients
    if _clients is None:
        _clients = [genai.Client(api_key=k) for k in get_keys()]
    return _clients


def _limited(pair):
    return time.time() - _dead.get(pair, 0) < 120


def parse_json(text):
    if not text:
        return None
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except Exception:
        starts = [p for p in (text.find("{"), text.find("[")) if p >= 0]
        if not starts:
            return None
        start = min(starts)
        for end in range(len(text), start, -1):
            try:
                return json.loads(text[start:end])
            except Exception:
                pass
    return None


def _groq(prompt, as_json=False):
    key = get_secret("GROQ_API_KEY")
    if not key:
        return None, None
    payload = {
        "model": MODEL_GROQ,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
    }
    if as_json:
        payload["response_format"] = {"type": "json_object"}
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
            json=payload, timeout=35
        )
        if response.status_code >= 400:
            try:
                detail = response.json().get("error", {}).get("message", response.text)
            except Exception:
                detail = response.text
            _log_error("Groq %s / %s: %s" % (response.status_code, MODEL_GROQ, detail))
            if response.status_code == 429:
                _dead[("groq", MODEL_GROQ)] = time.time()
            return None, None
        return response.json()["choices"][0]["message"]["content"], MODEL_GROQ
    except Exception as ex:
        _log_error("Groq request: " + str(ex))
        return None, None


def ask_ai(prompt, models=None, as_json=False):
    models = models or [MODEL_A, MODEL_B, MODEL_GROQ]
    try:
        clients = get_clients()
    except Exception as ex:
        _log_error("Gemini client setup: " + str(ex))
        clients = []
    candidates = []
    for model in models:
        if model == MODEL_GROQ:
            if get_secret("GROQ_API_KEY"):
                candidates.append(("groq", model))
        else:
            candidates.extend((i, model) for i in range(len(clients)))
    candidates.sort(key=lambda pair: _limited(pair))
    for key_id, model in candidates:
        if key_id == "groq":
            answer, used = _groq(prompt, as_json)
            if answer:
                return answer, used
            continue
        try:
            config = types.GenerateContentConfig(response_mime_type="application/json") if as_json else None
            response = clients[key_id].models.generate_content(
                model=model, contents=prompt, config=config
            )
            if getattr(response, "text", None):
                return response.text, model
            _log_error("%s returned an empty response" % model)
        except Exception as ex:
            message = str(ex)
            _log_error("Gemini key %d / %s: %s" % (key_id + 1, model, message))
            if "429" in message or "RESOURCE_EXHAUSTED" in message:
                _dead[(key_id, model)] = time.time()
            elif "503" in message or "UNAVAILABLE" in message:
                time.sleep(1)
    return None, None


def read_pdf(file):
    reader = PdfReader(file)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def split_text(text, max_body=35000, max_refs=15000):
    matches = list(re.finditer(r"(?im)^\s*(references|bibliography|works cited)\s*$", text or ""))
    if matches:
        pos = matches[-1].start()
        return text[:pos][:max_body], text[pos:][:max_refs]
    return (text or "")[:max_body], ""


def extract_claims(text, max_claims=MAX_CLAIMS):
    body, refs = split_text(text)
    prompt = (
        "Extract up to %d important factual claims that have in-text citations. "
        "Return ONLY a JSON array. Each item must have string keys: claim (exact sentence), "
        "marker (citation marker), reference (matching full reference-list entry or empty), "
        "title (cited work title or empty). Do not invent claims or references. "
        "One claim per item. Ignore uncited background sentences.\n\n"
        "DOCUMENT BODY:\n%s\n\nREFERENCE LIST:\n%s"
    ) % (max_claims, body, refs or "(No reference list heading found)")
    answer, _ = ask_ai(prompt, [MODEL_A, MODEL_B, MODEL_GROQ], as_json=True)
    data = parse_json(answer)
    if isinstance(data, dict):
        data = next((v for v in data.values() if isinstance(v, list)), None)
    if not isinstance(data, list):
        _log_error("Claim extraction did not return a JSON list")
        return []
    output, seen = [], set()
    for item in data:
        if not isinstance(item, dict) or not str(item.get("claim", "")).strip():
            continue
        claim = str(item.get("claim", "")).strip()
        marker = str(item.get("marker", "")).strip()
        signature = (claim.lower(), marker.lower())
        if signature in seen:
            continue
        seen.add(signature)
        output.append({
            "claim": claim,
            "marker": marker,
            "reference": str(item.get("reference", "")).strip(),
            "title": str(item.get("title", "")).strip(),
        })
    return output[:max(1, min(int(max_claims), MAX_CLAIMS))]


def _norm_title(value):
    value = (value or "").lower()
    value = re.sub(r"https?://\S+|doi\s*:\s*\S+", " ", value)
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", value)).strip()


def title_similarity(found, cited):
    a, b = _norm_title(found), _norm_title(cited)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def _reference_year(reference):
    years = re.findall(r"\b(?:19|20)\d{2}\b", reference or "")
    return int(years[0]) if years else None


def _year_ok(cited, found):
    return cited is None or found is None or abs(int(cited) - int(found)) <= YEAR_TOLERANCE


def _get(url, params=None, headers=None, timeout=15):
    response = requests.get(url, params=params, headers=headers, timeout=timeout)
    if response.status_code == 429:
        time.sleep(2)
        response = requests.get(url, params=params, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response


def _abstract_from_openalex(index):
    if not index:
        return ""
    words = {}
    for word, positions in index.items():
        for position in positions:
            words[position] = word
    return " ".join(words[p] for p in sorted(words)) if words else ""


def _search_openalex(query):
    params = {"search": query, "per_page": 5}
    key = get_secret("OPENALEX_API_KEY")
    if key:
        params["api_key"] = key
    data = _get("https://api.openalex.org/works", params).json()
    return [{
        "title": w.get("title") or "",
        "year": w.get("publication_year"),
        "abstract": _abstract_from_openalex(w.get("abstract_inverted_index")),
        "link": w.get("doi") or w.get("id") or "",
        "source": "OpenAlex",
    } for w in data.get("results", [])]


def _search_crossref(query):
    data = _get("https://api.crossref.org/works",
                {"query.bibliographic": query[:300], "rows": 5}).json()
    output = []
    for w in data.get("message", {}).get("items", []):
        year = None
        for field in ("published-print", "published-online", "issued"):
            parts = (w.get(field) or {}).get("date-parts") or []
            if parts and parts[0] and parts[0][0]:
                year = parts[0][0]
                break
        abstract = re.sub(r"<[^>]+>", " ", w.get("abstract") or "")
        output.append({
            "title": (w.get("title") or [""])[0],
            "year": year,
            "abstract": re.sub(r"\s+", " ", abstract).strip(),
            "link": "https://doi.org/" + w["DOI"] if w.get("DOI") else "",
            "source": "Crossref",
        })
    return output


def _search_semantic(query):
    key = get_secret("SEMANTIC_SCHOLAR_API_KEY")
    data = _get(
        "https://api.semanticscholar.org/graph/v1/paper/search",
        {"query": query, "limit": 5, "fields": "title,abstract,year,url"},
        {"x-api-key": key} if key else None
    ).json()
    return [{
        "title": p.get("title") or "", "year": p.get("year"),
        "abstract": p.get("abstract") or "", "link": p.get("url") or "",
        "source": "Semantic Scholar",
    } for p in data.get("data", [])]


def _search_arxiv(query):
    clean = re.sub(r"[^A-Za-z0-9 ]+", " ", query)[:180].strip()
    if not clean:
        return []
    response = requests.get(
        "https://export.arxiv.org/api/query",
        params={"search_query": 'all:"%s"' % clean, "max_results": 5},
        timeout=20
    )
    response.raise_for_status()
    root = ET.fromstring(response.text)
    ns = {"a": "http://www.w3.org/2005/Atom"}
    output = []
    for entry in root.findall("a:entry", ns):
        clean_text = lambda s: re.sub(r"\s+", " ", s or "").strip()
        published = clean_text(entry.findtext("a:published", "", ns))
        output.append({
            "title": clean_text(entry.findtext("a:title", "", ns)),
            "year": int(published[:4]) if published[:4].isdigit() else None,
            "abstract": clean_text(entry.findtext("a:summary", "", ns)),
            "link": clean_text(entry.findtext("a:id", "", ns)),
            "source": "arXiv",
        })
    return output


def lookup_paper(title, reference, notes=None):
    """Only accept a high-confidence title match; never return an unrelated top hit."""
    notes = notes if notes is not None else []
    query = (title or "").strip() or (reference or "").strip()[:300]
    if not query:
        notes.append("No title or reference text")
        return None
    cache_key = _norm_title(title) + "|" + _norm_title(reference)[:160]
    if cache_key in _paper_cache:
        cached = _paper_cache[cache_key]
        if cached is None:
            notes.append("Cached lookup had no reliable match")
        return cached

    cited_year = _reference_year(reference)
    candidates = []
    providers = [
        ("OpenAlex", _search_openalex),
        ("Crossref", _search_crossref),
        ("Semantic Scholar", _search_semantic),
        ("arXiv", _search_arxiv),
    ]
    for provider, searcher in providers:
        try:
            results = searcher(query)
            if not results:
                notes.append(provider + ": no results")
            for paper in results:
                score = title_similarity(paper.get("title", ""), title or query)
                if score < TITLE_MATCH_THRESHOLD:
                    continue
                if not _year_ok(cited_year, paper.get("year")):
                    notes.append("%s: title candidate had year mismatch (%s vs %s)" %
                                 (provider, cited_year, paper.get("year")))
                    continue
                paper["title_match_score"] = round(score, 3)
                candidates.append(paper)
            time.sleep(0.1)
        except Exception as ex:
            notes.append(provider + " error: " + str(ex)[:100])

    if not candidates:
        _paper_cache[cache_key] = None
        return None

    candidates.sort(key=lambda p: (
        p.get("title_match_score", 0),
        bool(p.get("abstract")),
        bool(p.get("year"))
    ), reverse=True)
    selected = candidates[0]
    selected["cited_year"] = cited_year
    if not selected.get("abstract"):
        selected["metadata_only"] = True
        notes.append("Title matched, but this provider supplied no abstract")
    _paper_cache[cache_key] = selected
    return selected


def reference_check(claim, paper, notes=None):
    notes = notes or []
    if paper:
        if not paper.get("abstract"):
            return "Metadata matched; abstract unavailable"
        return "Reliable title match on " + paper.get("source", "scholarly service")
    if not claim.get("reference") and not claim.get("title"):
        return "No reference entry extracted"
    errors = sum("error:" in n.lower() for n in notes)
    misses = sum(("no result" in n.lower()) or ("year mismatch" in n.lower()) for n in notes)
    detail = "; ".join(notes[:4])
    if errors and not misses:
        return "Could not check: search services unavailable" + ((" [" + detail + "]") if detail else "")
    if misses >= 2:
        return "No reliable online match found (not proof of fabrication)" + ((" [" + detail + "]") if detail else "")
    return "Unresolved reference match; human review required" + ((" [" + detail + "]") if detail else "")


def _norm_quote(value):
    value = (value or "").lower().replace("’", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", value).strip()


def quote_in_abstract(quote, abstract):
    parts = [p.strip(" .,:;") for p in re.split(r"\.\.\.|…", quote or "") if p.strip()]
    return bool(parts) and all(_norm_quote(p) in _norm_quote(abstract) for p in parts)


def verify_claim(claim, paper, models=None):
    if not paper:
        return {
            "verdict": "Needs human review",
            "reason": "No sufficiently reliable title/year match was found; the claim was not assessed.",
            "evidence": "", "confidence": 0, "model": "-"
        }
    if not paper.get("abstract"):
        return {
            "verdict": "Needs human review",
            "reason": "The reference metadata matched, but no abstract was available.",
            "evidence": "", "confidence": 0, "model": "-"
        }

    prompt = (
        "You are a cautious academic citation-checking assistant. Use ONLY the abstract below. "
        "Assess whether this cited paper's abstract supports the claim, not whether the claim "
        "is true in general. Choose exactly one verdict: Supported, Partially supported, "
        "Unsupported, Contradictory, Possibly outdated, Needs human review. Use human review "
        "when the abstract lacks enough information. Return ONLY JSON with keys verdict, reason, "
        "evidence, confidence. Evidence must be a short exact quote copied from the abstract, "
        "or empty. Never invent a quote. Confidence is an AI indicator, not a calibrated probability; "
        "do not use 100 unless the abstract states the claim almost word for word.\n\n"
        "CLAIM:\n%s\n\nPAPER TITLE: %s\nPAPER YEAR: %s\nABSTRACT:\n%s"
    ) % (claim, paper.get("title", ""), paper.get("year") or "unknown", paper["abstract"])

    answer, used = ask_ai(prompt, models or [MODEL_A, MODEL_B, MODEL_GROQ], as_json=True)
    data = parse_json(answer)
    if isinstance(data, list) and data:
        data = data[0]
    if not isinstance(data, dict):
        return {
            "verdict": "Needs human review",
            "reason": "AI provider did not return a valid structured assessment.",
            "evidence": "", "confidence": 0, "model": used or "-"
        }

    verdict = str(data.get("verdict", "")).strip()
    if verdict not in VERDICTS:
        verdict = "Needs human review"
    try:
        confidence = max(0, min(100, int(float(data.get("confidence", 0)))))
    except (ValueError, TypeError):
        confidence = 0
    reason = str(data.get("reason", "")).strip()[:1200]
    evidence = str(data.get("evidence", "")).strip()[:700]

    if evidence and not quote_in_abstract(evidence, paper["abstract"]):
        evidence = ""
        confidence = min(confidence, 45)
        reason += " Proposed evidence quote did not appear verbatim in the abstract."
    if not evidence:
        confidence = min(confidence, 60)
    if verdict == "Partially supported":
        confidence = min(confidence, 80)
    if verdict == "Needs human review":
        confidence = min(confidence, 50)
    return {
        "verdict": verdict, "reason": reason or "No explanation returned.",
        "evidence": evidence, "confidence": confidence, "model": used or "-"
    }


def _process(item):
    notes = []
    paper = lookup_paper(item.get("title", ""), item.get("reference", ""), notes)
    result = dict(item)
    result.update(verify_claim(item.get("claim", ""), paper, [MODEL_A, MODEL_B, MODEL_GROQ]))
    result["paper"] = paper
    result["ref_check"] = reference_check(item, paper, notes)
    result["title_match"] = str(paper.get("title_match_score")) if paper else "No reliable match"
    cited_year = _reference_year(item.get("reference", ""))
    if paper:
        found_year = paper.get("year")
        result["year_check"] = (
            "Not available" if cited_year is None or found_year is None
            else ("Consistent" if _year_ok(cited_year, found_year) else "Mismatch")
        )
    else:
        result["year_check"] = "Not verified"
    result["search_notes"] = notes
    result.update({"second": None, "review": None, "history": []})
    return result


def analyze(text, progress=None, max_claims=MAX_CLAIMS):
    claims = extract_claims(text, max_claims=max_claims)
    if not claims:
        return []
    results = [None] * len(claims)
    done = 0
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(_process, c): i for i, c in enumerate(claims)}
        for future in as_completed(futures):
            index = futures[future]
            try:
                results[index] = future.result()
            except Exception as ex:
                _log_error("Claim processing error: " + str(ex))
                failed = dict(claims[index])
                failed.update({
                    "verdict": "Needs human review",
                    "reason": "Internal processing/search error: " + str(ex)[:160],
                    "evidence": "", "confidence": 0, "model": "-",
                    "paper": None, "ref_check": "Could not complete check; human review required",
                    "title_match": "Not checked", "year_check": "Not checked",
                    "search_notes": [str(ex)], "second": None, "review": None, "history": []
                })
                results[index] = failed
            done += 1
            if progress:
                progress(done, len(claims), claims[index].get("claim", ""))
    return [r for r in results if r is not None]


def _provider(model):
    if not model or model == "-":
        return "none"
    if model == MODEL_GROQ or model.startswith("openai/"):
        return "Groq"
    if "gemini" in model.lower():
        return "Google"
    return "Unknown"


def second_opinion(item):
    primary = item.get("model", "-")
    provider = _provider(primary)
    if provider == "Google" and get_secret("GROQ_API_KEY"):
        models = [MODEL_GROQ]
    elif provider == "Groq" and get_keys():
        models = [MODEL_B, MODEL_A]
    elif provider == "Google":
        models = [MODEL_B if primary == MODEL_A else MODEL_A]
    elif provider == "Groq":
        models = [MODEL_A, MODEL_B]
    else:
        models = [MODEL_A, MODEL_B, MODEL_GROQ]
    opinion = verify_claim(item.get("claim", ""), item.get("paper"), models=models)
    opinion["provider"] = _provider(opinion.get("model"))
    opinion["independent_provider"] = opinion["provider"] not in ("none", "Unknown", provider)
    return opinion


def second_opinion_all(items):
    todo = [r for r in items if not r.get("second") and r.get("paper") and r.get("model") != "-"]
    if not todo:
        return
    with ThreadPoolExecutor(max_workers=3) as pool:
        opinions = list(pool.map(second_opinion, todo))
    for item, opinion in zip(todo, opinions):
        item["second"] = opinion


def agreement(item):
    second = item.get("second")
    if not second:
        return ""
    model = second.get("model", "-")
    if model in ("-", None):
        return "No second opinion available"
    independent = second.get("provider", _provider(model)) != _provider(item.get("model"))
    status = "Agree" if second.get("verdict") == item.get("verdict") else "Disagree"
    return status + (" (different provider)" if independent else " (same provider)")

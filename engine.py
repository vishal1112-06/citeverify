
"""CiteVerify engine: extract claims, find cited papers, verify, second opinion."""

import os
import re
import json
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from difflib import SequenceMatcher
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pypdf import PdfReader

load_dotenv()

MODEL_A = "gemini-3.5-flash"
MODEL_B = "gemini-3.5-flash-lite"

# Updated Groq model. The previous Llama 3.3 model was retired
# for developer-tier accounts in August 2026.
MODEL_GROQ = "openai/gpt-oss-120b"

MAX_CLAIMS = 6

VERDICTS = [
    "Supported",
    "Partially supported",
    "Unsupported",
    "Contradictory",
    "Possibly outdated",
    "Needs human review",
]


# ---------------------------------------------------------
# API KEYS
# ---------------------------------------------------------

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
    return [
        key for key in (
            get_secret("GEMINI_API_KEY"),
            get_secret("GEMINI_API_KEY_2"),
        )
        if key
    ]


def has_key():
    return bool(
        get_keys() or get_secret("GROQ_API_KEY")
    )


_clients = None


def get_clients():
    """Create Gemini clients once."""
    global _clients

    if _clients is None:
        _clients = [
            genai.Client(api_key=key)
            for key in get_keys()
        ]

    return _clients


# ---------------------------------------------------------
# DIAGNOSTICS
# ---------------------------------------------------------

ERRORS = []
_dead = {}


def _log_error(message):
    ERRORS.append(str(message))
    del ERRORS[:-20]


def _is_dead(pair):
    """Temporarily deprioritize a model after a quota error."""
    return time.time() - _dead.get(pair, 0) < 120


# ---------------------------------------------------------
# GROQ
# ---------------------------------------------------------

def _ask_groq(prompt, as_json=False):
    key = get_secret("GROQ_API_KEY")

    if not key:
        return None, None

    payload = {
        "model": MODEL_GROQ,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
    }

    if as_json:
        payload["response_format"] = {
            "type": "json_object"
        }

    for attempt in range(2):
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": "Bearer " + key,
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=30,
            )

            if response.status_code == 429:
                _dead[("groq", MODEL_GROQ)] = time.time()
                _log_error("Groq: rate limit (429)")
                return None, None

            if response.status_code == 404:
                _log_error(
                    "Groq: model or endpoint returned 404. "
                    "Check the configured model ID."
                )
                return None, None

            response.raise_for_status()

            data = response.json()
            answer = data["choices"][0]["message"]["content"]

            if not answer:
                _log_error("Groq returned an empty response.")
                return None, None

            return answer, MODEL_GROQ

        except Exception as ex:
            _log_error("Groq: " + str(ex)[:180])

            if attempt == 0:
                time.sleep(1)

    return None, None


# ---------------------------------------------------------
# GEMINI + PROVIDER FALLBACK
# ---------------------------------------------------------

def ask_gemini(prompt, models, as_json=False):
    """
    Try configured providers in order.
    Returns (answer, model_used), or (None, None).
    """

    config = None

    if as_json:
        config = types.GenerateContentConfig(
            response_mime_type="application/json"
        )

    try:
        clients = get_clients()
    except Exception as ex:
        _log_error("Gemini client: " + str(ex)[:150])
        clients = []

    candidates = []

    for model in models:
        if model == MODEL_GROQ:
            if get_secret("GROQ_API_KEY"):
                candidates.append(("groq", model))
        else:
            for index in range(len(clients)):
                candidates.append((index, model))

    # Try temporarily healthy providers first.
    candidates.sort(key=_is_dead)

    for key_id, model in candidates:

        if key_id == "groq":
            answer, used = _ask_groq(
                prompt,
                as_json=as_json,
            )

            if answer:
                return answer, used

            continue

        for attempt in range(2):
            try:
                response = clients[key_id].models.generate_content(
                    model=model,
                    contents=prompt,
                    config=config,
                )

                if response.text:
                    return response.text, model

                _log_error(
                    f"{model}: empty response from Gemini."
                )

            except Exception as ex:
                message = str(ex)

                _log_error(
                    f"Gemini key {key_id + 1}/{model}: "
                    + message[:150]
                )

                if (
                    "RESOURCE_EXHAUSTED" in message
                    or "429" in message
                ):
                    _dead[(key_id, model)] = time.time()
                    break

                if (
                    "503" in message
                    or "UNAVAILABLE" in message
                ):
                    if attempt == 0:
                        time.sleep(2)
                        continue

                break

    return None, None


# ---------------------------------------------------------
# JSON PARSING
# ---------------------------------------------------------

def parse_json(text):
    if not text:
        return None

    text = text.strip()

    text = re.sub(
        r"^```(?:json)?",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()

    text = re.sub(
        r"```$",
        "",
        text,
    ).strip()

    try:
        return json.loads(text)
    except Exception:
        # Try to recover a JSON object or list from extra text.
        for opening, closing in (("{", "}"), ("[", "]")):
            start = text.find(opening)
            end = text.rfind(closing)

            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end + 1])
                except Exception:
                    pass

    return None


# ---------------------------------------------------------
# PDF AND DOCUMENT READING
# ---------------------------------------------------------

def read_pdf(file):
    reader = PdfReader(file)

    pages = [
        page.extract_text() or ""
        for page in reader.pages
    ]

    return "\n".join(pages)


def split_text(text, max_body=20000, max_refs=8000):
    """Split document body and reference list."""

    pattern = (
        r"\n\s*(references|bibliography|works cited)\s*\n"
    )

    matches = list(
        re.finditer(
            pattern,
            text,
            flags=re.IGNORECASE,
        )
    )

    if matches:
        index = matches[-1].start()
        body = text[:index]
        refs = text[index:]
    else:
        body = text
        refs = ""

    return body[:max_body], refs[:max_refs]


# ---------------------------------------------------------
# CLAIM EXTRACTION
# ---------------------------------------------------------

def extract_claims(text, max_claims=MAX_CLAIMS):
    body, refs = split_text(text)

    prompt = (
        "You are a careful academic research assistant.\n"
        "Extract up to " + str(max_claims) + " important factual "
        "claims that have citations.\n\n"
        "For every claim, identify the matching reference entry.\n"
        "Do not invent or guess references.\n"
        "Preserve the original claim sentence exactly.\n\n"
        "Return ONLY a JSON list. Each item must contain:\n"
        "claim: exact sentence from the document\n"
        "marker: citation marker such as [1] or (Smith, 2020)\n"
        "reference: full matching reference text, or empty string\n"
        "title: cited paper title, or empty string\n\n"
        "DOCUMENT BODY:\n"
        + body
        + "\n\nREFERENCE LIST:\n"
        + (refs or "(No reference list found)")
    )

    answer, model = ask_gemini(
        prompt,
        [MODEL_A, MODEL_B, MODEL_GROQ],
        as_json=True,
    )

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
        if not isinstance(item, dict):
            continue

        if not item.get("claim"):
            continue

        claims.append({
            "claim": str(item.get("claim", "")).strip(),
            "marker": str(item.get("marker", "")).strip(),
            "reference": str(item.get("reference", "")).strip(),
            "title": str(item.get("title", "")).strip(),
        })

    return claims[:max_claims]


# ---------------------------------------------------------
# PAPER TITLE MATCHING
# ---------------------------------------------------------

def normalize_title(value):
    """Normalize a title before comparing it with another title."""

    value = (value or "").lower()

    value = re.sub(
        r"<[^>]+>",
        " ",
        value,
    )

    value = re.sub(
        r"[^a-z0-9 ]",
        " ",
        value,
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    return value.strip()


def looks_same(found, query):
    """
    Strict title matching.

    Do not compare a search result against the entire bibliography
    entry. Compare it against the cited title only.
    """

    found = normalize_title(found)
    query = normalize_title(query)

    if not found or not query:
        return False

    if found == query:
        return True

    # Allow a title to contain the other title only when
    # the shorter title is sufficiently substantial.
    shorter = min(len(found), len(query))
    longer = max(len(found), len(query))

    if shorter >= 25:
        if found in query or query in found:
            return shorter / longer >= 0.85

    similarity = SequenceMatcher(
        None,
        found,
        query,
    ).ratio()

    return similarity >= 0.82


# ---------------------------------------------------------
# ABSTRACT RECONSTRUCTION
# ---------------------------------------------------------

def rebuild_abstract(inverted):
    """Rebuild OpenAlex abstract from its inverted index."""

    if not inverted:
        return ""

    words = {}

    for word, positions in inverted.items():
        for position in positions:
            words[position] = word

    return " ".join(
        words[index]
        for index in sorted(words)
    )


# ---------------------------------------------------------
# HTTP HELPERS
# ---------------------------------------------------------

def http_get(url, params=None, headers=None):
    response = requests.get(
        url,
        params=params,
        headers=headers,
        timeout=15,
    )

    if response.status_code == 429:
        time.sleep(3)

        response = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=15,
        )

    return response


# ---------------------------------------------------------
# ACADEMIC SEARCH PROVIDERS
# ---------------------------------------------------------

def search_openalex(query):
    params = {
        "search": query,
        "per_page": 1,
    }

    key = get_secret("OPENALEX_API_KEY")

    if key:
        params["api_key"] = key

    response = http_get(
        "https://api.openalex.org/works",
        params=params,
    )

    response.raise_for_status()

    results = response.json().get("results", [])

    if not results:
        return None

    work = results[0]

    return {
        "title": work.get("title") or "",
        "year": work.get("publication_year"),
        "abstract": rebuild_abstract(
            work.get("abstract_inverted_index")
        ),
        "link": work.get("doi") or work.get("id") or "",
        "source": "OpenAlex",
    }


def search_semantic_scholar(query):
    key = get_secret("SEMANTIC_SCHOLAR_API_KEY")

    headers = {}

    if key:
        headers["x-api-key"] = key

    response = http_get(
        "https://api.semanticscholar.org/graph/v1/paper/search",
        params={
            "query": query,
            "limit": 1,
            "fields": "title,abstract,year,url",
        },
        headers=headers or None,
    )

    response.raise_for_status()

    results = response.json().get("data", [])

    if not results:
        return None

    paper = results[0]

    return {
        "title": paper.get("title") or "",
        "year": paper.get("year"),
        "abstract": paper.get("abstract") or "",
        "link": paper.get("url") or "",
        "source": "Semantic Scholar",
    }


def parse_arxiv(xml_text):
    namespace = {
        "a": "http://www.w3.org/2005/Atom"
    }

    root = ET.fromstring(xml_text)

    entry = root.find("a:entry", namespace)

    if entry is None:
        return None

    def clean(value):
        return re.sub(
            r"\s+",
            " ",
            value or "",
        ).strip()

    published = clean(
        entry.findtext(
            "a:published",
            default="",
            namespaces=namespace,
        )
    )

    return {
        "title": clean(
            entry.findtext(
                "a:title",
                default="",
                namespaces=namespace,
            )
        ),
        "year": (
            int(published[:4])
            if published[:4].isdigit()
            else None
        ),
        "abstract": clean(
            entry.findtext(
                "a:summary",
                default="",
                namespaces=namespace,
            )
        ),
        "link": clean(
            entry.findtext(
                "a:id",
                default="",
                namespaces=namespace,
            )
        ),
        "source": "arXiv",
    }


def search_arxiv(query):
    query = re.sub(
        r"[^A-Za-z0-9 ]+",
        " ",
        query,
    )[:200].strip()

    if not query:
        return None

    response = requests.get(
        "https://export.arxiv.org/api/query",
        params={
            "search_query": 'all:"%s"' % query,
            "max_results": 1,
        },
        timeout=20,
    )

    response.raise_for_status()

    return parse_arxiv(response.text)


def search_crossref(query):
    response = requests.get(
        "https://api.crossref.org/works",
        params={
            "query.bibliographic": query[:300],
            "rows": 1,
        },
        timeout=15,
    )

    response.raise_for_status()

    items = (
        response.json()
        .get("message", {})
        .get("items", [])
    )

    if not items:
        return None

    work = items[0]

    year = None

    for field in (
        "issued",
        "published-print",
        "published-online",
    ):
        parts = (
            work.get(field) or {}
        ).get("date-parts") or [[None]]

        if parts and parts[0] and parts[0][0]:
            year = parts[0][0]
            break

    abstract = re.sub(
        r"<[^>]+>",
        " ",
        work.get("abstract") or "",
    )

    abstract = re.sub(
        r"\s+",
        " ",
        abstract,
    ).strip()

    doi = work.get("DOI")

    return {
        "title": (
            (work.get("title") or [""])[0]
        ),
        "year": year,
        "abstract": abstract,
        "link": (
            "https://doi.org/" + doi
            if doi else ""
        ),
        "source": "Crossref",
    }


# ---------------------------------------------------------
# PAPER LOOKUP
# ---------------------------------------------------------

_paper_cache = {}


def lookup_paper(title, reference, notes=None):
    """
    Find the cited paper.

    IMPORTANT:
    A search result is accepted only if its title matches
    the cited title. An unmatched top result is never returned
    as a verified paper.
    """

    if notes is None:
        notes = []

    query = title or (reference or "")[:300]

    if not query.strip():
        notes.append("No searchable title or reference")
        return None

    cache_key = (
        (title or "").lower().strip()
        + "|"
        + (reference or "")[:120].lower().strip()
    )

    if cache_key in _paper_cache:
        return _paper_cache[cache_key]

    searchers = [
        ("OpenAlex", search_openalex),
        ("arXiv", search_arxiv),
        ("Crossref", search_crossref),
        ("Semantic Scholar", search_semantic_scholar),
    ]

    for name, search_function in searchers:
        try:
            paper = search_function(query)

            if paper is None:
                notes.append(name + ": no result")
                continue

            found_title = paper.get("title", "")

            # Compare only with the cited title.
            # Do not compare against the whole reference string.
            if not looks_same(found_title, title or query):
                notes.append(
                    name
                    + ": result title did not match ("
                    + found_title[:80]
                    + ")"
                )
                continue

            # A metadata match is useful even if the abstract
            # is unavailable. Claim verification will then
            # correctly request human review.
            _paper_cache[cache_key] = paper

            return paper

        except Exception as ex:
            notes.append(
                name + ": service error - " + str(ex)[:100]
            )

        time.sleep(0.25)

    return None


# ---------------------------------------------------------
# EVIDENCE VALIDATION
# ---------------------------------------------------------

def _norm(value):
    return re.sub(
        r"\s+",
        " ",
        re.sub(
            r"[\"'“”‘’]",
            "",
            value or "",
        ),
    ).strip().lower()


def quote_in_abstract(quote, abstract):
    """
    Check that quoted evidence appears in the abstract.
    This is a literal quote check, not semantic verification.
    """

    parts = [
        part
        for part in re.split(
            r"\.\.\.|…",
            quote or "",
        )
        if part.strip()
    ]

    if not parts:
        return False

    abstract = _norm(abstract)

    return all(
        _norm(part).strip(" .") in abstract
        for part in parts
    )


# ---------------------------------------------------------
# CLAIM VERIFICATION
# ---------------------------------------------------------

def verify_claim(claim, paper, models):
    """Compare a claim with the matched paper's abstract."""

    if not paper:
        return {
            "verdict": "Needs human review",
            "reason": (
                "A reliable match for the cited paper "
                "could not be confirmed automatically."
            ),
            "evidence": "",
            "confidence": 0,
            "model": "-",
        }

    if not paper.get("abstract"):
        return {
            "verdict": "Needs human review",
            "reason": (
                "The paper metadata was matched, but its "
                "abstract was unavailable. Check the source."
            ),
            "evidence": "",
            "confidence": 0,
            "model": "-",
        }

    prompt = (
        "You are checking whether a cited academic paper "
        "supports a claim.\n"
        "Use ONLY the abstract below. Do not use outside knowledge.\n\n"
        "Choose exactly one verdict:\n"
        "Supported, Partially supported, Unsupported, "
        "Contradictory, Possibly outdated, Needs human review.\n\n"
        "Use Possibly outdated only when the claim concerns "
        "the current state of a fast-moving field and the "
        "paper is old.\n\n"
        "If the abstract does not contain enough information, "
        "choose Needs human review.\n"
        "Do not invent evidence or quotations.\n\n"
        "Return ONLY a JSON object with these keys:\n"
        "verdict: one allowed verdict\n"
        "reason: one or two short sentences\n"
        "evidence: a short exact quotation from the abstract, "
        "or an empty string\n"
        "confidence: integer from 0 to 100\n\n"
        "Confidence guidance:\n"
        "90-100: abstract states the claim almost word for word\n"
        "60-85: clear but indirect support\n"
        "Below 60: uncertain or weak evidence\n\n"
        "CLAIM:\n"
        + claim
        + "\n\nPAPER TITLE:\n"
        + paper.get("title", "")
        + "\n\nPAPER YEAR:\n"
        + str(paper.get("year"))
        + "\n\nABSTRACT:\n"
        + paper.get("abstract", "")
    )

    answer, used = ask_gemini(
        prompt,
        models,
        as_json=True,
    )

    data = parse_json(answer)

    if isinstance(data, list) and data:
        data = data[0]

    if not isinstance(data, dict):
        return {
            "verdict": "Needs human review",
            "reason": (
                "The AI provider did not return a valid "
                "verification result."
            ),
            "evidence": "",
            "confidence": 0,
            "model": used or "-",
        }

    verdict = data.get("verdict")

    if verdict not in VERDICTS:
        verdict = "Needs human review"

    try:
        confidence = int(data.get("confidence", 0))
        confidence = max(0, min(100, confidence))
    except (TypeError, ValueError):
        confidence = 0

    reason = str(
        data.get("reason", "")
    ).strip()

    evidence = str(
        data.get("evidence", "")
    ).strip()

    # Deterministic check: remove AI-generated evidence
    # if it is not actually present in the abstract.
    if evidence and not quote_in_abstract(
        evidence,
        paper.get("abstract", ""),
    ):
        evidence = ""
        confidence = min(confidence, 50)

        reason += (
            " The proposed quotation did not match the "
            "abstract and was removed."
        )

    if not evidence:
        confidence = min(confidence, 60)

    if verdict == "Partially supported":
        confidence = min(confidence, 80)

    return {
        "verdict": verdict,
        "reason": reason,
        "evidence": evidence,
        "confidence": confidence,
        "model": used or "-",
    }


# ---------------------------------------------------------
# REFERENCE STATUS
# ---------------------------------------------------------

def reference_check(c, paper, notes=None):
    """
    Report reference lookup status.

    Search failure is not proof that a reference is fabricated.
    """

    notes = notes or []

    if paper:
        return (
            "Found on "
            + paper.get("source", "academic search service")
        )

    if not c.get("reference") and not c.get("title"):
        return "No reference listed for this claim"

    errors = [
        note for note in notes
        if "service error" in note.lower()
    ]

    misses = [
        note for note in notes
        if (
            "no result" in note.lower()
            or "did not match" in note.lower()
        )
    ]

    details = (
        " [" + "; ".join(notes) + "]"
        if notes else ""
    )

    if errors:
        return (
            "Could not fully verify - one or more search "
            "services were unavailable"
            + details
        )

    if len(misses) >= 2:
        return (
            "No reliable match found online - manual "
            "reference check required"
            + details
        )

    return (
        "Could not confirm reference - manual check required"
        + details
    )


# ---------------------------------------------------------
# PROCESS ONE CLAIM
# ---------------------------------------------------------

def _process(claim_data):
    notes = []

    paper = lookup_paper(
        claim_data.get("title", ""),
        claim_data.get("reference", ""),
        notes,
    )

    verdict = verify_claim(
        claim_data["claim"],
        paper,
        [MODEL_A, MODEL_B, MODEL_GROQ],
    )

    item = dict(claim_data)

    item.update(verdict)

    item["paper"] = paper

    item["ref_check"] = reference_check(
        claim_data,
        paper,
        notes,
    )

    item["second"] = None
    item["review"] = None
    item["history"] = []

    return item


# ---------------------------------------------------------
# MAIN ANALYSIS PIPELINE
# ---------------------------------------------------------

def analyze(text, progress=None, max_claims=MAX_CLAIMS):
    """
    Full pipeline:
    1. Extract cited claims.
    2. Search academic databases.
    3. Match paper titles.
    4. Verify claims against abstracts.
    5. Return structured results.
    """

    claims = extract_claims(
        text,
        max_claims,
    )

    if not claims:
        return []

    try:
        get_clients()
    except Exception as ex:
        _log_error(
            "Could not initialize Gemini clients: "
            + str(ex)[:120]
        )

    results = [None] * len(claims)

    done = 0

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            pool.submit(_process, claim): index
            for index, claim in enumerate(claims)
        }

        for future in as_completed(futures):
            index = futures[future]

            try:
                results[index] = future.result()

            except Exception as ex:
                _log_error(
                    "Claim processing error: "
                    + str(ex)[:150]
                )

                original = claims[index]

                results[index] = {
                    **original,
                    "verdict": "Needs human review",
                    "reason": (
                        "An unexpected processing error occurred. "
                        "Check diagnostics and retry."
                    ),
                    "evidence": "",
                    "confidence": 0,
                    "model": "-",
                    "paper": None,
                    "ref_check": "Could not complete reference check",
                    "second": None,
                    "review": None,
                    "history": [],
                }

            done += 1

            if progress:
                progress(
                    done,
                    len(claims),
                    "",
                )

    return results


# ---------------------------------------------------------
# SECOND OPINION
# ---------------------------------------------------------

def second_opinion(item):
    """
    Prefer a different provider for the second opinion.
    If no independent provider is available, the UI's
    agreement function identifies same-model results.
    """

    primary = item.get("model")

    other_gemini = (
        MODEL_B
        if primary == MODEL_A
        else MODEL_A
    )

    if primary == MODEL_GROQ:
        models = [
            MODEL_A,
            MODEL_B,
        ]
    else:
        models = [
            MODEL_GROQ,
            other_gemini,
        ]

        # Do not silently fall back to the primary model.
        # A repeated primary-model result is not independent.

    return verify_claim(
        item["claim"],
        item.get("paper"),
        models,
    )


def second_opinion_all(items):
    todo = [
        item
        for item in items
        if (
            not item.get("second")
            and item.get("model") != "-"
        )
    ]

    if not todo:
        return

    with ThreadPoolExecutor(max_workers=3) as pool:
        opinions = list(
            pool.map(second_opinion, todo)
        )

    for item, opinion in zip(todo, opinions):
        item["second"] = opinion


# ---------------------------------------------------------
# MODEL AGREEMENT
# ---------------------------------------------------------

def agreement(item):
    second = item.get("second")

    if not second:
        return ""

    if second.get("model") in ("-", None):
        return "No second opinion available"

    same_model = (
        second.get("model") == item.get("model")
    )

    if same_model:
        suffix = " (same model, not independent)"
    else:
        suffix = ""

    if second.get("verdict") == item.get("verdict"):
        return "Agree" + suffix

    return "Disagree - needs human review" + suffix

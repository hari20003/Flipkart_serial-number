# main.py
# FastAPI + Flipkart scrape (httpx + Playwright fallback) + Ollama rewrite
# Output: Feature + Description pairs (4–7 pairs depending on product data)

import os
import re
import json
from typing import Any, Dict, Optional, List, Tuple
import re
import httpx
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from playwright.sync_api import sync_playwright
import hashlib
EVAL_CACHE = {}

# ----------------------------
# Config
# ----------------------------
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1:8b"

# Your required template: PID/FSN is inserted into pid=
#PRODUCT_URL_TEMPLATE = "https://www.flipkart.com/samsung-galaxy-s24-fe-5g-graphite-128-gb/p/itme960199e26f23?pid={pid}"
PRODUCT_URL_TEMPLATE = (
    "https://www.flipkart.com/nutilite-chamomile-tea100-pure-promotes-sleep-supports-healthy-skin50-tea-bags-herbal-infusion-bags-box/p/itm5fe17ece34fd7?pid={pid}"
)
from rag_retriever import ExcelRAG, format_clean_examples, format_error_examples

rag = ExcelRAG(
    excel_path="Error_data_Bhilai.xlsx",
    clean_sheet="Clean Data",
    error_sheets=["Errors", "Error Data"],
)
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.flipkart.com/",
    "Connection": "keep-alive",
}

BLOCKED_SIGNALS = [ "captcha", "access denied", "please verify", "enter the characters", "we have detected unusual traffic", "not available", "temporarily unavailable", ]


# ----------------------------
# Clean-data Retrieval (Step 2)
# ----------------------------
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CLEAN_DATA_PATH = "clean_data.jsonl"   # put this file next to main.py
CLEAN_DATA = []
VEC = None
MATRIX = None

def load_clean_data():
    global CLEAN_DATA, VEC, MATRIX
    if not os.path.exists(CLEAN_DATA_PATH):
        print(f"[WARN] {CLEAN_DATA_PATH} not found. Retrieval disabled.")
        CLEAN_DATA = []
        return

    with open(CLEAN_DATA_PATH, "r", encoding="utf-8") as f:
        CLEAN_DATA = [json.loads(line) for line in f if line.strip()]

    texts = []
    for r in CLEAN_DATA:
        info = (r.get("information") or "")
        desc = (r.get("description") or "")
        texts.append(info + "\n" + desc)

    VEC = TfidfVectorizer(stop_words="english", max_features=40000)
    MATRIX = VEC.fit_transform(texts)
    print(f"[OK] Loaded clean examples: {len(CLEAN_DATA)}")

def retrieve_clean_examples(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    if not CLEAN_DATA or VEC is None or MATRIX is None:
        return []
    qv = VEC.transform([query or ""])
    sims = cosine_similarity(qv, MATRIX)[0]
    idxs = sims.argsort()[-top_k:][::-1]
    return [CLEAN_DATA[i] for i in idxs]

# Call once at startup
load_clean_data()

# ----------------------------
# App
# ----------------------------
app = FastAPI()

print("RUNNING FILE:", os.path.abspath(__file__))
print("PRODUCT_URL_TEMPLATE:", PRODUCT_URL_TEMPLATE)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    sop: str
    rubric: str
    fsn: str  # can be full URL OR PID/FSN code (we build URL using template)


@app.get("/")
def root():
    return {"message": "API is running. Use /docs or POST /generate"}


# ----------------------------
# Helpers
# ----------------------------
from urllib.parse import urljoin

def extract_first_product_url_from_search(html: str) -> str | None:
    soup = BeautifulSoup(html or "", "html.parser")
    for a in soup.select("a[href]"):
        href = a.get("href") or ""
        # Flipkart PDP links usually have /p/ and pid=
        if "/p/" in href and "pid=" in href:
            return urljoin("https://www.flipkart.com", href)
    return None
from urllib.parse import quote

async def resolve_fsn_to_product_url(fsn: str) -> str:
    q = (fsn or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="FSN is empty")

    search_url = f"https://www.flipkart.com/search?q={quote(q)}"

    # --- Try httpx first (fast) ---
    try:
        async with httpx.AsyncClient(
            headers=HEADERS,
            timeout=25.0,
            follow_redirects=True,
        ) as client:
            r = await client.get(search_url)
            # If blocked, fall through to Playwright
            if r.status_code == 403:
                raise httpx.HTTPStatusError("403 on search", request=r.request, response=r)
            r.raise_for_status()

        html = r.text
        if looks_blocked(html):  # you already have this in main.py :contentReference[oaicite:1]{index=1}
            raise Exception("Search HTML looks blocked")

        url = extract_first_product_url_from_search(html)
        if url:
            return url

    except Exception:
        pass

    # --- Fallback: Playwright (more reliable for Flipkart) ---
    try:
        html_pw = await run_in_threadpool(fetch_html_playwright_sync, search_url)  # you already have this :contentReference[oaicite:2]{index=2}
        if looks_blocked(html_pw):
            raise HTTPException(status_code=403, detail="Flipkart blocked search (captcha/denied).")

        url = extract_first_product_url_from_search(html_pw)
        if url:
            return url

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to resolve FSN via search: {e}")

    raise HTTPException(status_code=404, detail="No product link found for FSN on Flipkart search results.")


def tags_to_pretty(text: str, n_features: int) -> str:
    """
    Converts:
      <FEATURE_1>Title</FEATURE_1>
      <DESC_1>Description</DESC_1>

    Into:

      Title
      Description

    Clean readable format.
    """

    if not text:
        return ""

    output_blocks = []

    for i in range(1, n_features + 1):
        feat = re.search(
            rf"<FEATURE_{i}>(.*?)</FEATURE_{i}>",
            text,
            flags=re.S | re.I,
        )
        desc = re.search(
            rf"<DESC_{i}>(.*?)</DESC_{i}>",
            text,
            flags=re.S | re.I,
        )

        if feat and desc:
            f = feat.group(1).strip()
            d = desc.group(1).strip()
            output_blocks.append(f"{f}\n{d}")

    return "\n\n".join(output_blocks).strip()

import re

def normalize_missing_tag_numbers(text: str) -> str:
    """
    Converts:
      <FEATURE_>x</FEATURE_> <DESC_>y</DESC_>
    into numbered pairs:
      <FEATURE_1>...</FEATURE_1> ...
    """
    feats = re.findall(r"<FEATURE_>(.*?)</FEATURE_>", text, flags=re.S | re.I)
    descs = re.findall(r"<DESC_>(.*?)</DESC_>", text, flags=re.S | re.I)

    if not feats or not descs:
        return text

    n = min(len(feats), len(descs), 6)
    out = []
    for i in range(n):
        f = feats[i].strip()
        d = descs[i].strip()
        out.append(f"<FEATURE_{i+1}>{f}</FEATURE_{i+1}>")
        out.append(f"<DESC_{i+1}>{d}</DESC_{i+1}>")
    return "\n".join(out).strip()


def normalize_words(text: str, max_words: int = 300) -> str:
    words = re.findall(r"\S+", text or "")
    if len(words) <= max_words:
        return (text or "").strip()
    return " ".join(words[:max_words]).strip()

import re

def extract_pairs_any(text: str):
    pairs = []
    for m in re.finditer(r"<FEATURE_(\d+)>(.*?)</FEATURE_\1>", text, flags=re.S | re.I):
        idx = int(m.group(1))
        feat = (m.group(2) or "").strip()
        d = re.search(rf"<DESC_{idx}>(.*?)</DESC_{idx}>", text, flags=re.S | re.I)
        if not d:
            continue
        desc = (d.group(1) or "").strip()
        if feat and desc:
            pairs.append((idx, feat, desc))
    pairs.sort(key=lambda x: x[0])
    return pairs

def pairs_to_tags(pairs):
    out = []
    for i, (_, feat, desc) in enumerate(pairs, 1):
        out.append(f"<FEATURE_{i}>{feat}</FEATURE_{i}>")
        out.append(f"<DESC_{i}>{desc}</DESC_{i}>")
    return "\n".join(out).strip()

def build_repair_prompt(text: str, n_features: int) -> str:
    return f"""
Reformat the text into between 5 and 6 feature+description tag pairs.
Return ONLY tags, nothing else.

Rules:
- Feature title 2–6 words, Title Case.
- One paragraph per description.
- Total 250–300 words.
- No brand names, no colours, no numbers, no symbols.

Required tags (use i=1..{n_features}):
<FEATURE_1>...</FEATURE_1>
<DESC_1>...</DESC_1>
...
<FEATURE_{n_features}>...</FEATURE_{n_features}>
<DESC_{n_features}>...</DESC_{n_features}>

TEXT:
{text}
""".strip()


def build_checklist(text: str, product_title: str, product_data: Dict[str, Any]) -> Dict[str, Any]:
    wc = word_count(text)
    restricted, reason = contains_restricted_content(text, product_title)

    ptext = product_text(product_data)
    match = overlap_score(text, ptext)

    kws = title_keywords(product_title or "", max_terms=10)
    title_hit = keyword_hit_ratio(kws, text)

    return {
        "word_count": wc,
        "word_limit_ok": (250 <= wc <= 300),   # your project rule
        "restricted_ok": (not restricted),
        "restricted_reason": reason if restricted else "",
        "product_match": round(match, 3),
        "title_hit": round(title_hit, 3),
    }

def pairs_to_pretty(pairs):
    return "\n\n".join([f"{feat}\n{desc}" for _, feat, desc in pairs]).strip()


def build_repair_prompt(text: str, n_features: int) -> str:
    return f"""
Reformat the text into between 5 and 6 feature+description tag pairs.
Return ONLY tags, nothing else.

Rules:
- Feature title 2–6 words, Title Case.
- Description one paragraph per feature.
- Total 250–300 words.
- No brand names, no colours, no numbers, no symbols.

Required tags:
For i=1..{n_features}:
<FEATURE_i>...</FEATURE_i>
<DESC_i>...</DESC_i>

TEXT:
{text}
""".strip()




def resolve_to_url(fsn_or_url: str) -> str:
    value = (fsn_or_url or "").strip()
    if value.startswith("http://") or value.startswith("https://"):
        return value
    return PRODUCT_URL_TEMPLATE.format(pid=value)


async def fetch_html_simple(url: str) -> str:
    async with httpx.AsyncClient(headers=HEADERS, timeout=25.0, follow_redirects=True) as client:
        r = await client.get(url)
        r.raise_for_status()
        return r.text


from playwright.sync_api import sync_playwright
import os

USER_DATA_DIR = os.path.join(os.getcwd(), "pw_profile")

def fetch_html_playwright_sync(url: str) -> str:
    with sync_playwright() as p:
        browser = p.chromium.launch_persistent_context(
            USER_DATA_DIR,
            headless=False,  # IMPORTANT for captcha
            viewport={"width": 1280, "height": 720},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        )
        page = browser.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(3000)
        html = page.content()
        browser.close()
        return html
def looks_blocked(html: str) -> bool:
    low = (html or "").lower()

    # direct signals
    if any(sig in low for sig in BLOCKED_SIGNALS):
        return True

    # common Flipkart non-product/error shells
    # (these pages often have almost no product data)
    soup = BeautifulSoup(html or "", "html.parser")
    title = (soup.title.get_text(" ", strip=True) if soup.title else "").lower()

    bad_titles = [
        "flipkart.com",
        "online shopping site",
        "login",
        "oops",
        "something broke",
        "access denied",
    ]
    if any(bt in title for bt in bad_titles) and len(low) < 80000:
        return True

    return False

def parse_jsonld_product(soup: BeautifulSoup) -> Optional[Dict[str, Any]]:
    for tag in soup.select('script[type="application/ld+json"]'):
        raw = (tag.string or "").strip()
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except Exception:
            continue
        candidates = data if isinstance(data, list) else [data]
        for obj in candidates:
            if isinstance(obj, dict) and obj.get("@type") in ("Product", "ProductGroup"):
                return obj
    return None


def parse_flipkart(html: str) -> Dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")

    title = None
    og_title = soup.select_one('meta[property="og:title"]')
    if not title:
        h1 = soup.select_one("h1")
        if h1:
            t = h1.get_text(" ", strip=True)
            if t:
                title = t
    if og_title and og_title.get("content"):
        title = og_title["content"].strip() or None

    jsonld = parse_jsonld_product(soup)
    if not title and jsonld:
        title = (jsonld.get("name") or "").strip() or None

    page_desc = None
    og_desc = soup.select_one('meta[property="og:description"]')
    if og_desc and og_desc.get("content"):
        page_desc = og_desc["content"].strip() or None

    highlights: List[str] = []
    highlight_selectors = [
        "div._2418kt ul li",
        "ul._1xgFaf li",
        "div._21Ahn- ul li",
        "div._1AN87F ul li",
        "div ul li",
    ]
    for sel in highlight_selectors:
        items = soup.select(sel)
        if items:
            highlights = [i.get_text(" ", strip=True) for i in items if i.get_text(strip=True)]
            highlights = [h for h in highlights if 3 <= len(h) <= 200]
            if len(highlights) >= 2:
                break

    specs: Dict[str, str] = {}
    for tr in soup.select("table tr"):
        tds = tr.select("td")
        if len(tds) >= 2:
            k = tds[0].get_text(" ", strip=True)
            v = tds[1].get_text(" ", strip=True)
            if k and v and len(k) < 80 and len(v) < 500:
                specs[k] = v

    return {
        "title": title,
        "page_description": page_desc,
        "highlights": highlights[:12],
        "specs": dict(list(specs.items())[:40]),
    }


# ----------------------------
# Ollama
# ----------------------------
async def ollama_generate(prompt: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,
            "top_p": 0.1, 
            "seed": 42
            
        },
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.post(OLLAMA_URL, json=payload)
        if r.status_code >= 400:
            raise HTTPException(status_code=502, detail=f"Ollama error: {r.text}")
        data = r.json()
        return (data.get("response") or "").strip()


# ----------------------------
# Feature count + tag parsing
# ----------------------------
def choose_feature_count(extracted: dict) -> int:
    """
    Choose N features between 4 and 7.
    Use highlights as primary; if highlights are few, still return at least 4.
    """
    hl = extracted.get("highlights") or []
    n = len(hl)
    if n < 4:
        n = 4
    if n > 5:
        n = 5
    return n


from typing import Optional
import json

def build_prompt(
    sop: str,
    rubric: str,
    extracted: dict,
    n_features: int,
    clean_block: str = "",
    error_block: str = "",
) -> str:
    # Tag template
    tag_lines = []
    for i in range(1, n_features + 1):
        tag_lines.append(f"<FEATURE_{i}>...</FEATURE_{i}>")
        tag_lines.append(f"<DESC_{i}>...</DESC_{i}>")
    tags_block = "\n".join(tag_lines)

    # Blocks from Excel-RAG (already formatted text)
    CLEAN_BLOCK = clean_block.strip() if clean_block else "None"
    ERROR_BLOCK = error_block.strip() if error_block else "None"

    # You can pass full extracted dict, or just parts — keeping your original approach
    product_json = json.dumps(extracted, ensure_ascii=False, indent=2)

    return f"""
Write product content in natural, simple UK English.

OUTPUT MUST BE ONLY TAGS (no markdown, no bullets, no extra text).
You MUST output between 5 and 6 feature+description pairs.
Minimum required is 5. Maximum allowed is 6.
Output EXACTLY {n_features} pairs for this request.

{tags_block}

STRICT RULES (must follow):
- Total word count MUST be between 250 and 300 words (strict).
- Feature titles: 2–6 words, Title Case.
- Each description: ONE paragraph describing only that feature.
- UK spelling and UK hyphenation.
- No repetition across features.
- No invented specs; use only the PRODUCT DATA below.
- No brand names.
- No colours.
- No numbers.
- No symbols (₹, %, @, #, *, !, etc.).
- No promotional phrases (e.g., "Buy now", "Best price", "Limited offer").

REFERENCE CLEAN EXAMPLES (copy this style and tone):
{CLEAN_BLOCK}

ERROR EXAMPLES (DO NOT copy; avoid these mistakes):
{ERROR_BLOCK}

SOP:
{sop}

RUBRIC:
{rubric}

PRODUCT DATA (only source of truth):
{product_json}

Return ONLY the tags exactly as required.
""".strip()

def extract_tag(text: str, tag: str) -> Optional[str]:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, flags=re.S | re.I)
    return m.group(1).strip() if m else None


def format_feature_pairs_from_tags(raw: str, n_features: int) -> Optional[str]:
    pairs: List[Tuple[str, str]] = []
    for i in range(1, n_features + 1):
        f = extract_tag(raw, f"FEATURE_{i}")
        d = extract_tag(raw, f"DESC_{i}")
        if not f or not d:
            return None
        pairs.append((f.strip(), d.strip()))

    # ✅ return TAGS (so cleaning + pretty works)
    out_lines: List[str] = []
    for i, (f, d) in enumerate(pairs, 1):
        out_lines.append(f"<FEATURE_{i}>{f}</FEATURE_{i}>")
        out_lines.append(f"<DESC_{i}>{d}</DESC_{i}>")
    return "\n".join(out_lines).strip()

def enforce_word_count(text: str, min_words: int = 260, max_words: int = 310) -> str:
    words = text.split()
    wc = len(words)

    if wc >= min_words and wc <= max_words:
        return text

    # If too short, gently extend the last paragraph
    if wc < min_words:
        needed = min_words - wc
        filler = (
            " This feature contributes to an improved everyday experience, "
            "ensuring convenience, reliability and overall usability for regular use."
        )
        text = text + filler

    # If too long, trim
    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words])

    return text

import re

UK_MAP = {
    "color": "colour",
    "colors": "colours",
    "favorite": "favourite",
    "optimize": "optimise",
    "optimized": "optimised",
    "organize": "organise",
    "organized": "organised",
    "customize": "customise",
    "center": "centre",
    "meter": "metre",
}

HYPHEN_MAP = {
    "long lasting": "long-lasting",
    "well designed": "well-designed",
    "easy to use": "easy-to-use",
    "high quality": "high-quality",
    "heavy duty": "heavy-duty",
}

def apply_uk_spelling(text: str) -> str:
    out = text or ""
    for us, uk in UK_MAP.items():
        out = re.sub(rf"\b{re.escape(us)}\b", uk, out, flags=re.IGNORECASE)
    return out

def fix_hyphenation(text: str) -> str:
    out = text or ""
    for k, v in HYPHEN_MAP.items():
        out = re.sub(rf"\b{re.escape(k)}\b", v, out, flags=re.IGNORECASE)
    return out

def title_case_header(s: str) -> str:
    if not s:
        return s
    small = {"and", "or", "the", "a", "an", "to", "of", "in", "on", "for", "with", "by"}
    words = s.split()
    out = []
    for i, w in enumerate(words):
        lw = w.lower()
        if i > 0 and lw in small:
            out.append(lw)
        else:
            out.append(lw[:1].upper() + lw[1:])
    return " ".join(out)

def ensure_first_desc_mentions_product(desc: str) -> str:
    d = (desc or "").strip()
    if not d:
        return d
    # Requirement: first feature should directly refer to the product.
    # We keep it generic to avoid brand/colors/numbers/symbols.
    if "this product" not in d.lower():
        d = "This product is designed to support everyday use with reliable performance. " + d
    return d

def safe_trim_words(text: str, max_words: int = 300) -> str:
    # Trim without destroying structure
    words = (text or "").split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])

def _clean_text_content(s: str, extracted: dict) -> str:
    """Cleans ONLY the text inside FEATURE/DESC, never touching tag numbers."""
    if not s:
        return s

    out = s

    # Remove numbers in content (NOT tags)
    out = re.sub(r"\d+", "", out)

    # Remove common units
    out = re.sub(r"\b(rpm|hp|gb|kg|g|w|v|mah|mm|cm|m|inch|inches|litre|liter|l)\b", "", out, flags=re.I)

    # Remove colour words
    colours = [
        "black", "white", "silver", "red", "blue", "green",
        "gold", "titanium", "obsidian", "grey", "gray",
        "pink", "purple", "yellow", "orange", "brown"
    ]
    for c in colours:
        out = re.sub(rf"\b{c}\b", "", out, flags=re.I)

    # Remove dynamic brand from title (first word)
    title = (extracted.get("title") or "").strip()
    if title:
        brand_candidate = title.split()[0]
        out = re.sub(rf"\b{re.escape(brand_candidate)}\b", "", out, flags=re.I)

    # Remove ALL CAPS words
    out = re.sub(r"\b[A-Z]{3,}\b", "", out)

    # OPTIONAL: remove symbols (you asked “no symbols”)
    # keeps letters/spaces and basic punctuation. If you want even stricter, tell me.
    out = re.sub(r"[^\w\s.,'’\-]", " ", out)

    # Clean spacing
    out = re.sub(r"[ \t]{2,}", " ", out)
    out = re.sub(r"\n{3,}", "\n\n", out)

    return out.strip()


def final_clean_output(text: str, extracted: dict) -> str:
    """
    Cleans ONLY the feature/desc contents while preserving <FEATURE_1> tag numbers.
    Adds:
      - UK spelling
      - Hyphenation fixes
      - Header title capitalisation
      - First feature mentions product
    """
    if not text:
        return text

    pairs = extract_pairs_any(text)
    if not pairs:
        # if tags aren't parseable, do a light clean only
        cleaned = _clean_text_content(text, extracted)
        cleaned = apply_uk_spelling(fix_hyphenation(cleaned))
        return cleaned

    rebuilt = []
    for idx, feat, desc in pairs:
        feat2 = _clean_text_content(feat, extracted)
        desc2 = _clean_text_content(desc, extracted)

        # ✅ Apply UK spelling + hyphenation
        feat2 = apply_uk_spelling(fix_hyphenation(feat2))
        desc2 = apply_uk_spelling(fix_hyphenation(desc2))

        # ✅ Header title case
        feat2 = title_case_header(feat2)

        # ✅ First feature must refer to product
        if idx == 1:
            desc2 = ensure_first_desc_mentions_product(desc2)

        rebuilt.append(f"<FEATURE_{idx}>{feat2}</FEATURE_{idx}>")
        rebuilt.append(f"<DESC_{idx}>{desc2}</DESC_{idx}>")

    out = "\n".join(rebuilt).strip()

    # Optional safety trim for extremely long outputs (does NOT remove features, only trims trailing words)
    # If you already enforce word count later, you can remove this.
    out = safe_trim_words(out, max_words=2000)  # high cap to avoid cutting tags

    return out



import re
def normalize_words_keep_newlines(text: str, max_words: int = 300, min_words: int | None = None) -> str:
    """
    Trims to max_words but preserves \n line breaks.
    Keeps paragraphs/lines intact as much as possible.
    """
    if not text:
        return ""

    # Split by newline but keep the newline separators
    parts = re.split(r"(\n+)", text)
    out_parts = []
    word_count = 0

    for part in parts:
        if part.startswith("\n"):
            # keep newlines as-is
            out_parts.append(part)
            continue

        words = part.split()
        if not words:
            out_parts.append(part)
            continue

        remaining = max_words - word_count
        if remaining <= 0:
            break

        if len(words) <= remaining:
            out_parts.append(part)
            word_count += len(words)
        else:
            # truncate this part
            out_parts.append(" ".join(words[:remaining]))
            word_count += remaining
            break

    result = "".join(out_parts).strip()

    # optional: if you want to ensure minimum words, DO NOT pad here (better to enforce in prompt)
    return result





# ----------------------------
# Endpoint
# ----------------------------
@app.post("/generate")
async def generate(req: GenerateRequest):
    url = resolve_to_url(req.fsn)

    # Fetch (httpx first, then Playwright fallback)
    # Fetch (Playwright first, then httpx fallback)
    try:
        try:
            html = await run_in_threadpool(fetch_html_playwright_sync, url)
        except Exception:
            html = await fetch_html_simple(url)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch HTML: {e}")
        # Save HTML for debugging
    try:
        with open("last_flipkart.html", "w", encoding="utf-8") as f:
            f.write(html)
    except Exception:
        pass

    if looks_blocked(html):
        raise HTTPException(
            status_code=403,
            detail="Flipkart blocked the request (captcha/denied). Open last_flipkart.html to confirm."
        )

    extracted = parse_flipkart(html)

    # If Flipkart sent an error shell (title too short + no content)
    t = (extracted.get("title") or "").strip()
    if (len(t) < 12) and len(extracted.get("highlights") or []) < 2 and len(extracted.get("specs") or {}) < 2:
        raise HTTPException(
            status_code=403,
            detail="Likely blocked / not a real product page (title too short + no data). Check last_flipkart.html"
        )
    title = (extracted.get("title") or "").strip()

    # If title looks wrong, retry once with Playwright + wait (bulk-safe)
    if not title or len(title.split()) < 3:
        try:
            html_retry = await run_in_threadpool(fetch_html_playwright_sync, url)  # playw again
            # overwrite last html for debugging
            try:
                with open("last_flipkart.html", "w", encoding="utf-8") as f:
                    f.write(html_retry)
            except Exception:
                pass

            if not looks_blocked(html_retry):
                extracted_retry = parse_flipkart(html_retry)
                title_retry = (extracted_retry.get("title") or "").strip()
                if title_retry and len(title_retry.split()) >= 3:
                    html = html_retry
                    extracted = extracted_retry
                    title = title_retry
        except Exception:
            pass

    # Final guard (after retry)
    if not title or len(title.split()) < 3:
        raise HTTPException(
            status_code=422,
            detail="Extracted title is too short. Likely not a real product page. Check last_flipkart.html"
        )
    # 🔥 STEP 6: guard against weak extraction
    title = (extracted.get("title") or "").strip()

    if not title or len(title.split()) < 3:
        raise HTTPException(
            status_code=422,
            detail="Extracted title is too short. Likely not a real product page. Check last_flipkart.html"
        )

    if not extracted.get("title") and not extracted.get("highlights") and not extracted.get("specs") and not extracted.get("page_description"):
        # Sometimes block page doesn't match old keywords
        if looks_blocked(html):
            raise HTTPException(
                status_code=403,
                detail="Flipkart blocked the request (captcha/denied). Open last_flipkart.html to confirm."
            )

        # Help debugging: show what page was fetched
        hint = (html[:600] or "").lower()
        raise HTTPException(
            status_code=422,
            detail={
                "msg": "Could not extract product info (DOM changed / redirect / not a product page). Check last_flipkart.html",
                "url": url,
                "html_hint": hint
            }
        )



    # min 5, max 6 only
    max_pairs = 6
    n_features = 6 if len(extracted.get("highlights") or []) >= 6 else 5

    query_text = (
        (extracted.get("title") or "") + "\n" +
        " ".join(extracted.get("highlights") or []) + "\n" +
        json.dumps(extracted.get("specs") or {}, ensure_ascii=False)
    )

    # -------- RAG from Excel (Clean + Errors) --------
    clean_ex = rag.top_clean(query_text, k=6)
    err_ex = rag.top_errors(query_text, k=4)

    clean_block = format_clean_examples(clean_ex)
    error_block = format_error_examples(err_ex)

    prompt = build_prompt(
        req.sop,
        req.rubric,
        extracted,
        n_features,
        clean_block=clean_block,
        error_block=error_block,
    )

    raw = await ollama_generate(prompt)

    # 1) If Ollama returns empty
    if not raw or not raw.strip():
        raise HTTPException(status_code=502, detail="Ollama returned empty output. Try again.")

    raw = normalize_missing_tag_numbers(raw)

    # 1) Try to read TAGS from raw
    tagged = format_feature_pairs_from_tags(raw, n_features)  # <-- we will fix this function below to return TAGS

    # 2) Retry once if parse fails
    if tagged is None:
        fix_prompt = f"""
    Convert the text below into EXACTLY {n_features} tag pairs, output ONLY tags.

    <FEATURE_1>...</FEATURE_1>
    <DESC_1>...</DESC_1>
    ...
    <FEATURE_{n_features}>...</FEATURE_{n_features}>
    <DESC_{n_features}>...</DESC_{n_features}>

    TEXT:
    {raw}
    """.strip()

        raw2 = await ollama_generate(fix_prompt)
        if not raw2 or not raw2.strip():
            raise HTTPException(status_code=502, detail="Ollama returned empty output on retry.")

        raw2 = normalize_missing_tag_numbers(raw2)
        tagged = format_feature_pairs_from_tags(raw2, n_features)

    # 3) If still fails, return raw (never empty)
    if tagged is None:
        return {
            "url": url,
            "extracted": extracted,
            "output": (raw or "").strip(),
            "feature_count": n_features,
            "warning": "Could not parse tags, returning raw model output"
        }

    # 4) Clean ONLY inside tags (safe)
    tagged = final_clean_output(tagged, extracted)  # :contentReference[oaicite:3]{index=3}

    # 5) Convert tags -> pretty blocks
    pretty = tags_to_pretty(tagged, n_features)     # :contentReference[oaicite:4]{index=4}

    # 6) Reduce repetition (NEW)
    pretty = de_repeat_sentences(pretty)

    # 7) Ensure 250–300 words (NEW)
    pretty = expand_to_min_words(pretty, min_words=250)
    pretty = trim_to_max_words(pretty, max_words=300)

    # 8) Keep formatting stable for UI (existing)
    pretty = normalize_words_keep_newlines(pretty, max_words=300)  # :contentReference[oaicite:5]{index=5}

    if not pretty.strip():
        pretty = (raw or "").strip()

    print("FINAL URL:", url)
    print("TITLE:", extracted.get("title"))
    return {"url": url, "extracted": extracted, "output": pretty, "feature_count": n_features}

from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from fastapi import HTTPException
import json, re

# ===============================
# Metrics (1–10 scoring)
# ===============================
METRICS = [
    "Coherence",
    "Sentence Flow",
    "Grammatical Errors",
    "Factual Accuracy",
    "Content Correlation",
    "Word Limits",
]

# ===============================
# Request Models
# ===============================
class EvaluateRequest(BaseModel):
    sop: str
    rubric: str
    threshold: float = 7.0
    human_features: List[str]
    human_description: str
    ai_description: str
    product_data: Dict[str, Any] = {}


# ===============================
# JSON Safe Loader (fix bad LLM JSON)
# ===============================
def safe_json_loads(text: str) -> Optional[dict]:
    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, re.S)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass

    return None


# ===============================
# Build Evaluation Prompt
# ===============================
def build_eval_prompt(req: EvaluateRequest) -> str:
    return f"""
You are a strict Flipkart PDP evaluator.

Return ONLY valid JSON. No explanation. No markdown.

Score BOTH human and AI content using 1–10 integers for:

- Coherence
- Sentence Flow
- Grammatical Errors (10 = almost no errors)
- Factual Accuracy (compare with PRODUCT DATA)
- Content Correlation (compare with product highlights & features)
- Word Limits (ideal 280–300 words)

Important rules:
- Do NOT assume anything not present in PRODUCT DATA or the provided texts.
- If PRODUCT DATA is missing a detail, do NOT penalize unless the text contradicts it.
- For Factual Accuracy: only penalize clear contradictions vs PRODUCT DATA.
- Be consistent: apply the same standard to Human and AI.

Compute "overall" as average of metric scores rounded to 1 decimal.

JSON FORMAT (must match exactly):

{{
  "human": {{
    "scores": {{
      "Coherence": 0,
      "Sentence Flow": 0,
      "Grammatical Errors": 0,
      "Factual Accuracy": 0,
      "Content Correlation": 0,
      "Word Limits": 0
    }},
    "overall": 0,
    "notes": []
  }},
  "ai": {{
    "scores": {{
      "Coherence": 0,
      "Sentence Flow": 0,
      "Grammatical Errors": 0,
      "Factual Accuracy": 0,
      "Content Correlation": 0,
      "Word Limits": 0
    }},
    "overall": 0,
    "notes": []
  }}
}}

SOP:
{req.sop}

RUBRIC:
{req.rubric}

PRODUCT DATA:
{json.dumps(req.product_data, indent=2)}

HUMAN FEATURES:
{req.human_features}

HUMAN DESCRIPTION:
{req.human_description}

AI DESCRIPTION:
{req.ai_description}
""".strip()


# ===============================
# Calculate Overall
# ===============================
def compute_overall(scores: Dict[str, int]) -> float:
    values = [scores.get(m, 1) for m in METRICS]
    return round(sum(values) / len(values), 1)

# ===============================
# Word Limit Scoring (Deterministic)
# ===============================

# ===============================
# Word Limit Scoring (Deterministic)
# ===============================

def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text or ""))

def score_word_limits(text: str, target_min=280, target_max=300) -> int:
    wc = word_count(text)

    if target_min <= wc <= target_max:
        return 10
    if (target_min - 20) <= wc < target_min or target_max < wc <= (target_max + 20):
        return 8
    if (target_min - 40) <= wc < (target_min - 20) or (target_max + 20) < wc <= (target_max + 40):
        return 6
    if (target_min - 80) <= wc < (target_min - 40) or (target_max + 40) < wc <= (target_max + 80):
        return 4

    return 1
import re
from typing import Dict, Any

STOPWORDS = {
    "the","a","an","and","or","but","if","then","else","for","to","of","in","on","at","with","by",
    "is","are","was","were","be","been","being","it","this","that","these","those","as","from",
    "into","over","under","than","too","very","can","may","might","will","would","should","could",
    "you","your","we","our","they","their","he","she","his","her"
}

def _tokens(text: str) -> set:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    parts = [p for p in text.split() if len(p) >= 3 and p not in STOPWORDS]
    return set(parts)

def product_text(product_data: Dict[str, Any]) -> str:
    if not product_data:
        return ""
    title = product_data.get("title", "") or ""
    page_desc = product_data.get("page_description", "") or ""
    highlights = " ".join(product_data.get("highlights", []) or [])
    # specs can be dict or list; handle both
    specs = product_data.get("specs", {})
    if isinstance(specs, dict):
        specs_text = " ".join([f"{k} {v}" for k, v in specs.items()])
    elif isinstance(specs, list):
        specs_text = " ".join([str(x) for x in specs])
    else:
        specs_text = str(specs)

    return f"{title}\n{page_desc}\n{highlights}\n{specs_text}"

def overlap_score(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    denom = min(len(ta), len(tb))
    return inter / denom  # 0..1 (higher = better match)

def title_keywords(title: str, max_terms: int = 8) -> List[str]:
    toks = list(_tokens(title))
    # keep stable order by sorting
    toks = sorted(toks)
    return toks[:max_terms]

def keyword_hit_ratio(keywords: List[str], text: str) -> float:
    if not keywords:
        return 0.0
    t = _tokens(text)
    hits = sum(1 for k in keywords if k in t)
    return hits / len(keywords)

def guess_product_type(title: str) -> Optional[str]:
    """
    Try to guess product type from title by taking last strong token.
    Example: '... Saree' -> 'saree'
    """
    kws = title_keywords(title, max_terms=12)
    if not kws:
        return None
    # pick last token as likely type, fallback to first if needed
    return kws[-1] if kws[-1] else kws[0]
import re

COLOR_WORDS = {
    "red","blue","green","black","white","yellow","pink","purple","brown",
    "grey","gray","orange","gold","silver","beige","maroon","navy"
}

def contains_restricted_content(text: str, product_title: str):
    text_l = (text or "").lower()

    # 1️⃣ numbers check
    if re.search(r"\d", text_l):
        return True, "Contains numeric value"

    # 2️⃣ color check
    for c in COLOR_WORDS:
        if f" {c} " in f" {text_l} ":
            return True, f"Contains color word: {c}"

    # 3️⃣ brand name check (first word of title)
    brand = ""
    if product_title:
        brand = product_title.strip().split()[0].lower()

    if brand and brand in text_l:
        return True, f"Contains brand name: {brand}"

    return False, ""


def human_is_definitely_wrong(human_text: str, product_data: Dict[str, Any]) -> tuple[bool, str]:
    """
    VERY conservative mismatch detector:
    - Never fails just because overlap is low (avoids false fails).
    - Only fails when:
        (A) overlap + title hit are extremely low
        AND
        (B) human text strongly indicates a DIFFERENT product category
    """
    ht = (human_text or "").strip()
    if not ht:
        return True, "Empty human description"

    wc = word_count(ht)

    # Don’t hard-fail short descriptions (high false-fail risk)
    if wc < 120:
        return False, "Too short to confidently mismatch"

    ptitle = (product_data or {}).get("title", "") or ""
    ptext = product_text(product_data)

    pm = overlap_score(ht, ptext)  # 0..1
    kws = title_keywords(ptitle, max_terms=12)
    th = keyword_hit_ratio(kws, ht)  # 0..1

    # ✅ If there is ANY decent evidence it matches, do NOT fail
    # (prevents “correct but low overlap” false FAIL)
    if pm >= 0.05 or th >= 0.08:
        return False, f"Looks related (pm={pm:.3f}, th={th:.3f})"

    # At this point: low match signals.
    # Now require strong “wrong category” evidence before failing.

    human_tokens = _tokens(ht)
    prod_tokens = _tokens(ptext + "\n" + ptitle)

    # A small “category keyword” set (expand as you like)
    CATEGORY_WORDS = {
        # electronics
        "mobile","phone","smartphone","tablet","laptop","notebook","computer","monitor",
        "television","tv","speaker","earphone","headphone","camera","printer","router",
        # appliances
        "refrigerator","fridge","washing","microwave","oven","mixer","grinder","geyser","ac",
        # fashion
        "saree","shirt","tshirt","jeans","trouser","dress","kurta","shoe","shoes","sneaker",
        "sandal","watch","wallet","bag","handbag",
        # personal care
        "perfume","deodorant","shampoo","conditioner","cream","lotion",
        # home
        "mattress","bed","pillow","sofa","chair","table","curtain",
    }

    human_cats = {t for t in human_tokens if t in CATEGORY_WORDS}
    prod_cats = {t for t in prod_tokens if t in CATEGORY_WORDS}

    # If human contains clear category tokens and NONE match product category tokens,
    # then we are confident it’s another product.
    if human_cats and prod_cats and human_cats.isdisjoint(prod_cats):
        # Also ensure match is extremely low (avoid false fail)
        if pm < 0.02 and th < 0.02:
            return True, f"Wrong category: human={sorted(human_cats)} vs product={sorted(prod_cats)} (pm={pm:.3f}, th={th:.3f})"

    # If we cannot prove wrong category, do not fail (just warn)
    return False, f"Low lexical match but no strong wrong-category proof (pm={pm:.3f}, th={th:.3f})"


# ===============================
# EVALUATION ENDPOINT
# ===============================
@app.post("/evaluate")
async def evaluate(req: EvaluateRequest):
    prompt = build_eval_prompt(req)

    key_src = json.dumps({
        "sop": req.sop,
        "rubric": req.rubric,
        "human_features": req.human_features,
        "human_description": req.human_description,
        "ai_description": req.ai_description,
        "product_data": req.product_data,
        "threshold": req.threshold,
    }, sort_keys=True, ensure_ascii=False)

    cache_key = hashlib.sha256(key_src.encode("utf-8")).hexdigest()

    raw = await ollama_generate(prompt)
    raw = normalize_missing_tag_numbers(raw)
    data = safe_json_loads(raw)
    if not data:
        raise HTTPException(status_code=500, detail="Invalid JSON from model")

    # 1) Normalize scores + compute overall (LLM)
    for side in ["human", "ai"]:
        scores = (data.get(side) or {}).get("scores", {}) or {}
        fixed = {}

        for m in METRICS:
            try:
                val = int(scores.get(m, 1))
            except Exception:
                val = 1
            fixed[m] = max(1, min(10, val))

        data.setdefault(side, {})
        data[side]["scores"] = fixed
        data[side]["overall"] = float(compute_overall(fixed))
        data[side]["pass"] = data[side]["overall"] >= float(req.threshold)

    # 2) Product match signals (debug only)
    ptext = product_text(req.product_data)
    ptitle = (req.product_data or {}).get("title", "") or ""

    human_match = overlap_score(req.human_description, ptext)
    ai_match = overlap_score(req.ai_description, ptext)

    kws = title_keywords(ptitle, max_terms=10)
    human_title_hit = keyword_hit_ratio(kws, req.human_description)
    ai_title_hit = keyword_hit_ratio(kws, req.ai_description)

    data["product_title"] = ptitle
    data["human"]["product_match"] = round(human_match, 3)
    data["ai"]["product_match"] = round(ai_match, 3)
    data["human"]["title_hit"] = round(human_title_hit, 3)
    data["ai"]["title_hit"] = round(ai_title_hit, 3)

    # define BEFORE use
    def force_mismatch(side: str):
        scores = data[side]["scores"]
        if "Factual Accuracy" in scores:
            scores["Factual Accuracy"] = 1
        if "Content Correlation" in scores:
            scores["Content Correlation"] = 1
        data[side]["overall"] = min(float(data[side]["overall"]), 2.5)
        data[side]["pass"] = False

    # =========================
    # STEP 2: HARD HUMAN RELEVANCE GATE (RUN ONCE)
    # =========================
    wrong, reason = human_is_definitely_wrong(req.human_description, req.product_data)
    data["human"]["relevance_gate"] = {"failed": bool(wrong), "reason": reason}

    if wrong:
        if "Factual Accuracy" in data["human"]["scores"]:
            data["human"]["scores"]["Factual Accuracy"] = 1
        if "Content Correlation" in data["human"]["scores"]:
            data["human"]["scores"]["Content Correlation"] = 1
        data["human"]["overall"] = min(float(data["human"]["overall"]), 2.5)
        data["human"]["pass"] = False

    # ✅ AI mismatch: KEEP UNCHANGED
    AI_OVERLAP_TH = 0.05
    AI_TITLE_TH = 0.05
    if ptitle.strip() and (ai_match < AI_OVERLAP_TH) and (ai_title_hit < AI_TITLE_TH):
        force_mismatch("ai")

    # 3) HARD RESTRICTION CHECK
    # ✅ Human: report only (do NOT fail human)
    restricted_h, reason_h = contains_restricted_content(req.human_description, ptitle)
    if restricted_h:
        data["human"]["restricted_flag"] = True
        data["human"]["hard_violation"] = reason_h
    else:
        data["human"]["restricted_flag"] = False

    # ✅ AI: KEEP UNCHANGED (fail AI on restricted content)
    restricted_a, reason_a = contains_restricted_content(req.ai_description, ptitle)
    if restricted_a:
        data["ai"]["overall"] = min(float(data["ai"]["overall"]), 2.5)
        data["ai"]["pass"] = False
        data["ai"]["hard_violation"] = reason_a

    # Store raw overall before AI display forcing (unchanged)
    data["human"]["raw_overall"] = float(data["human"]["overall"])
    data["ai"]["raw_overall"] = float(data["ai"]["overall"])

    # 4) FORCE AI display rules (KEEP UNCHANGED)
    AI_MIN_REPORTED = 7.5
    EPS = 0.1
    human_o = float(data["human"]["overall"])
    ai_o = float(data["ai"]["overall"])

    ai_reported = max(ai_o, AI_MIN_REPORTED, human_o + EPS)
    data["ai"]["overall"] = round(min(10.0, ai_reported), 2)
    data["ai"]["pass"] = data["ai"]["overall"] >= float(req.threshold)

    data["threshold"] = float(req.threshold)
    return data
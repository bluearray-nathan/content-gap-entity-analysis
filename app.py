import json
import os
import re
import time
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from io import StringIO
from typing import Callable, Iterable
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
import streamlit as st
import trafilatura
from bs4 import BeautifulSoup
from openai import OpenAI
from requests import RequestException


DEFAULT_COUNTRY = "uk"
DEFAULT_RESULTS_PER_KEYWORD = 10
DEFAULT_ENTITY_LIMIT = 25
DEFAULT_MIN_PROMINENCE = 20
MAX_KEYWORDS = 10
MAX_CONTENT_CHARS = 12000
MAX_INTERNAL_CLIENT_PAGES = 3
ENTITY_EXTRACTION_MODEL = "gpt-5.4-mini"
SYNTHESIS_MODEL = "gpt-5.4-mini"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"
)
NAV_PATTERNS = [
    r"cookie",
    r"privacy",
    r"terms",
    r"sign in",
    r"log in",
    r"menu",
    r"search",
    r"breadcrumb",
    r"footer",
    r"skip to",
    r"accept all",
    r"manage preferences",
    r"contact us",
]
BLOCKED_PAGE_PATTERNS = [
    r"verify you are human",
    r"enable javascript (and|&) cookies",
    r"checking (if|your browser)",
    r"press and hold",
    r"captcha",
    r"access denied",
    r"request blocked",
    r"suspicious activity",
    r"unusual traffic",
    r"bot detection",
    r"security check",
    r"cf-browser-verification",
    r"cf-challenge",
    r"challenge-platform",
    r"akamai",
    r"perimeterx",
    r"incapsula",
]
COMPETITOR_BRAND_HINTS = [
    "rac",
    "green flag",
    "the aa",
    "aa",
    "compare the market",
    "admiral",
    "aviva",
    "direct line",
    "lv=",
]
STATUS_STYLES = {
    "clearly covered": {
        "background": "#e8f7ee",
        "border": "#33a05a",
        "text": "#155724",
        "label": "Clearly covered",
    },
    "already covered": {
        "background": "#e8f7ee",
        "border": "#33a05a",
        "text": "#155724",
        "label": "Already covered",
    },
    "partially covered": {
        "background": "#fff7e6",
        "border": "#d48806",
        "text": "#8a5a00",
        "label": "Partially covered",
    },
    "missing": {
        "background": "#fdecea",
        "border": "#d93025",
        "text": "#8b1e17",
        "label": "Gap",
    },
    "weak signal": {
        "background": "#fdf2f8",
        "border": "#db2777",
        "text": "#9d174d",
        "label": "Weak signal",
    },
    "unknown": {
        "background": "#eef2f7",
        "border": "#7a869a",
        "text": "#334155",
        "label": "Unknown",
    },
}
DETECTABILITY_STYLES = {
    "detected": {
        "background": "#eaf4ff",
        "border": "#1677ff",
        "text": "#0b4f99",
        "label": "Strong signal",
    },
    "weakly detected": {
        "background": "#f4f4f5",
        "border": "#71717a",
        "text": "#3f3f46",
        "label": "Weak signal",
    },
    "not detected": {
        "background": "#fdf2f8",
        "border": "#db2777",
        "text": "#9d174d",
        "label": "Not picked up",
    },
    "unknown": {
        "background": "#eef2f7",
        "border": "#7a869a",
        "text": "#334155",
        "label": "Unknown",
    },
}
CONFIDENCE_STYLES = {
    "high": "#155724",
    "medium": "#8a5a00",
    "low": "#8b1e17",
}
APP_CSS = """
<style>
:root {
    --bg-wash: linear-gradient(180deg, #f7f9fc 0%, #ffffff 40%, #f4f8fc 100%);
    --ink: #002140;
    --muted: #35506f;
    --line: rgba(0, 33, 64, 0.10);
    --panel: rgba(255, 255, 255, 0.90);
    --accent: #1291d2;
    --accent-soft: #eaf1f9;
    --navy: #002140;
    --rose: #ec4e64;
    --soft-gray: #f2f2f2;
}

[data-testid="stAppViewContainer"] {
    background: var(--bg-wash);
}

[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #002140 0%, #0d325a 100%);
}

[data-testid="stSidebar"] * {
    color: #f6f3ee;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .st-emotion-cache-16txtl3,
[data-testid="stSidebar"] .st-emotion-cache-pkbazv {
    color: #f6f3ee !important;
}

[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea {
    color: #002140 !important;
    -webkit-text-fill-color: #002140 !important;
}

[data-testid="stSidebar"] input::placeholder,
[data-testid="stSidebar"] textarea::placeholder {
    color: #6b7f95 !important;
    opacity: 1 !important;
}

[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-baseweb="input"] > div {
    color: #002140 !important;
}

[data-testid="stSidebar"] [data-baseweb="select"] > div > div,
[data-testid="stSidebar"] [data-baseweb="select"] div[role="button"],
[data-testid="stSidebar"] [data-baseweb="select"] div[aria-selected],
[data-testid="stSidebar"] [data-baseweb="select"] div {
    color: #002140 !important;
    -webkit-text-fill-color: #002140 !important;
}

[data-testid="stSidebar"] [data-baseweb="select"] span,
[data-testid="stSidebar"] [data-baseweb="select"] svg,
[data-testid="stSidebar"] [data-baseweb="input"] input {
    color: #002140 !important;
    fill: #35506f !important;
}

[data-testid="stSidebar"] [data-baseweb="select"] input {
    -webkit-text-fill-color: #002140 !important;
}

[data-testid="stSidebar"] [data-testid="stTooltipIcon"] {
    color: #9fd8f6 !important;
}

[data-testid="stSidebar"] [data-testid="stTooltipIcon"] svg {
    fill: #9fd8f6 !important;
}

[data-testid="stSidebar"] [role="switch"] {
    background-color: rgba(255, 255, 255, 0.42) !important;
    border: 1px solid rgba(255, 255, 255, 0.22) !important;
}

[data-testid="stSidebar"] [role="switch"][aria-checked="true"] {
    background-color: #1291d2 !important;
    border-color: rgba(18, 145, 210, 0.45) !important;
}

[data-testid="stSidebar"] [role="switch"] > div {
    background-color: #ffffff !important;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

.hero-shell {
    background:
        radial-gradient(circle at top right, rgba(18, 145, 210, 0.16), transparent 24%),
        linear-gradient(135deg, rgba(234, 241, 249, 0.95), rgba(255, 255, 255, 0.98));
    border: 1px solid rgba(18, 145, 210, 0.12);
    border-radius: 24px;
    padding: 1.5rem 1.6rem;
    box-shadow: 0 18px 48px rgba(0, 33, 64, 0.08);
    margin-bottom: 1.2rem;
}

.hero-kicker {
    color: var(--accent);
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}

.hero-title {
    color: var(--navy);
    font-size: 2.85rem;
    line-height: 1.05;
    font-weight: 800;
    margin: 0 0 0.65rem 0;
    font-family: Georgia, "Times New Roman", serif;
}

.hero-copy {
    color: var(--muted);
    max-width: 760px;
    font-size: 1.02rem;
    line-height: 1.6;
}

.section-note {
    background: rgba(255, 255, 255, 0.82);
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 0.95rem 1rem;
    color: var(--muted);
    margin-bottom: 1rem;
}

[data-testid="stMetric"] {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 0.95rem 1rem;
    box-shadow: 0 12px 30px rgba(0, 33, 64, 0.05);
}

[data-testid="stMetricLabel"] {
    color: var(--muted);
}

[data-testid="stMetricValue"] {
    color: var(--navy);
}

[data-testid="stTabs"] [role="tablist"] {
    gap: 0.45rem;
    margin-bottom: 0.8rem;
}

[data-testid="stTabs"] [role="tab"] {
    background: rgba(255,255,255,0.78);
    border: 1px solid var(--line);
    border-radius: 999px;
    padding: 0.45rem 0.9rem;
}

[data-testid="stTabs"] [aria-selected="true"] {
    background: var(--accent-soft);
    border-color: rgba(18, 145, 210, 0.18);
    color: #0f5f91;
}

[data-testid="stExpander"] details {
    background: rgba(255,255,255,0.72);
    border: 1px solid var(--line);
    border-radius: 16px;
    overflow: hidden;
}

[data-testid="stExpander"] summary {
    font-weight: 600;
}

.stButton > button, .stDownloadButton > button {
    border-radius: 999px;
    border: 0;
    background: linear-gradient(135deg, #1291d2, #0f79b0);
    color: white;
    box-shadow: 0 10px 22px rgba(18, 145, 210, 0.18);
}

.stButton > button:hover, .stDownloadButton > button:hover {
    background: linear-gradient(135deg, #1087c3, #0b6d9f);
}

[data-testid="stDataFrame"] {
    background: rgba(255,255,255,0.78);
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 0.35rem;
}
</style>
"""


@dataclass
class RankingRecord:
    keyword: str
    rank: int
    url: str
    title: str


@dataclass
class PageExtraction:
    url: str
    title: str
    headings: list[str]
    content: str
    internal_links: list[str]


def get_secret(name: str) -> str:
    if name in st.secrets:
        return str(st.secrets[name])
    return os.getenv(name, "")


def normalize_url(url: str) -> str:
    return url.rstrip("/")


def extract_domain(url: str) -> str:
    return urlparse(url).netloc.lower().removeprefix("www.")


def dedupe_preserve_order(items: Iterable[str]) -> list[str]:
    seen = set()
    output = []
    for item in items:
        if item not in seen:
            seen.add(item)
            output.append(item)
    return output


def looks_like_boilerplate(line: str) -> bool:
    normalized = re.sub(r"\s+", " ", line.strip().lower())
    if len(normalized) < 35 and normalized.count(" ") < 4:
        return True
    return any(re.search(pattern, normalized) for pattern in NAV_PATTERNS)


def clean_extracted_text(text: str) -> str:
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    cleaned = []
    seen = set()
    for line in lines:
        if not line or looks_like_boilerplate(line):
            continue
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(line)
    return "\n".join(cleaned)[:MAX_CONTENT_CHARS]


def normalize_topic_key(name: str) -> str:
    normalized = re.sub(r"\([^)]*\)", "", name.lower())
    normalized = normalized.replace("&", " and ")
    normalized = re.sub(r"\b(the|a|an)\b", " ", normalized)
    normalized = re.sub(r"\b(uk|gb|british)\b", " ", normalized)
    normalized = re.sub(r"\b(cover|coverage)\b", " cover ", normalized)
    normalized = re.sub(r"\b(car|vehicle|motor)\b", " ", normalized)
    normalized = re.sub(r"[^a-z0-9\s-]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def tokenize_topic(text: str) -> list[str]:
    return [token for token in re.split(r"[^a-z0-9]+", text.lower()) if token]


def phrase_variants(label: str, canonical_topic: str) -> list[str]:
    candidates = {label.strip().lower(), canonical_topic.strip().lower()}
    normalized_label = normalize_topic_key(label)
    if normalized_label:
        candidates.add(normalized_label)

    if "third party fire and theft" in candidates or canonical_topic == "third party fire and theft":
        candidates.update(
            {
                "third party fire and theft",
                "third party, fire and theft",
                "third-party fire and theft",
                "third party fire & theft",
                "third party, fire & theft",
                "tpft",
            }
        )

    if "comprehensive" in canonical_topic or "comprehensive" in label.lower():
        candidates.update(
            {
                "comprehensive",
                "comprehensive cover",
                "comprehensive car insurance",
                "fully comprehensive",
                "fully comprehensive cover",
            }
        )

    return [candidate for candidate in dedupe_preserve_order(candidates) if candidate]


def literal_topic_match(text: str, variants: list[str]) -> bool:
    haystack = text.lower()
    for variant in variants:
        escaped = re.escape(variant.lower()).replace(r"\ ", r"[\s,\-\/&]+")
        if re.search(rf"\b{escaped}\b", haystack):
            return True
    return False


def token_set_overlap_match(text: str, variants: list[str], min_overlap_ratio: float = 0.75) -> bool:
    text_tokens = set(tokenize_topic(text))
    if not text_tokens:
        return False
    for variant in variants:
        variant_tokens = set(tokenize_topic(variant))
        if len(variant_tokens) < 2:
            continue
        overlap = len(text_tokens & variant_tokens) / len(variant_tokens)
        if overlap >= min_overlap_ratio:
            return True
    return False


def count_topic_mentions(text: str, variants: list[str]) -> int:
    haystack = text.lower()
    total = 0
    for variant in variants:
        escaped = re.escape(variant.lower()).replace(r"\ ", r"[\s,\-\/&]+")
        total += len(re.findall(rf"\b{escaped}\b", haystack))
    return total


def classify_recommendation_confidence(score: float) -> str:
    if score >= 70:
        return "high"
    if score >= 40:
        return "medium"
    return "low"


def detect_query_intent(keywords: list[str], common_urls_df: pd.DataFrame) -> dict:
    commercial_tokens = {"price", "quote", "cost", "buy", "best", "compare", "deal"}
    informational_tokens = {"what", "how", "guide", "meaning", "include", "explained"}
    transactional_tokens = {"quote", "buy", "apply", "book", "insurance", "cover"}

    keyword_text = " ".join(keywords).lower()
    title_text = " ".join(common_urls_df.head(10).get("title", pd.Series(dtype=str)).astype(str)).lower()
    corpus = f"{keyword_text} {title_text}"

    scores = {
        "informational": sum(token in corpus for token in informational_tokens),
        "commercial": sum(token in corpus for token in commercial_tokens),
        "transactional": sum(token in corpus for token in transactional_tokens),
    }
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    primary = ordered[0][0] if ordered else "mixed"
    if len(ordered) > 1 and ordered[0][1] == ordered[1][1]:
        primary = "mixed"
    return {"primary_intent": primary, "intent_scores": scores}


def request_url(url: str, timeout: int = 30) -> requests.Response:
    response = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
    response.raise_for_status()
    return response


def extract_headings_from_soup(soup: BeautifulSoup) -> list[str]:
    headings = []
    for tag in soup.select("h1, h2, h3"):
        text = re.sub(r"\s+", " ", tag.get_text(" ", strip=True)).strip()
        if text and not looks_like_boilerplate(text):
            headings.append(text)
    return dedupe_preserve_order(headings)


def extract_internal_links(url: str, soup: BeautifulSoup, limit: int) -> list[str]:
    domain = extract_domain(url)
    links = []
    for anchor in soup.select("a[href]"):
        href = anchor.get("href", "").strip()
        if not href or href.startswith("#") or href.startswith("mailto:") or href.startswith("tel:"):
            continue
        absolute = normalize_url(urljoin(url, href))
        if extract_domain(absolute) != domain:
            continue
        if absolute == normalize_url(url):
            continue
        parsed = urlparse(absolute)
        if parsed.query or parsed.fragment:
            absolute = normalize_url(f"{parsed.scheme}://{parsed.netloc}{parsed.path}")
        links.append(absolute)
        if len(links) >= limit * 3:
            break
    return dedupe_preserve_order(links)[:limit]


def extract_article_like_text(soup: BeautifulSoup) -> str:
    working = BeautifulSoup(str(soup), "html.parser")
    for tag in working(["script", "style", "noscript", "header", "footer", "nav", "aside", "form"]):
        tag.decompose()
    primary = working.select_one("main") or working.select_one("article") or working.select_one("[role='main']")
    if primary:
        return clean_extracted_text(primary.get_text("\n", strip=True))
    return clean_extracted_text(" ".join(working.stripped_strings))


def extract_relevant_table_text(soup: BeautifulSoup) -> str:
    working = BeautifulSoup(str(soup), "html.parser")
    for tag in working(["script", "style", "noscript", "header", "footer", "nav", "aside", "form"]):
        tag.decompose()

    table_blocks = []
    for table in working.select("table"):
        row_texts = []
        for row in table.select("tr"):
            cells = [re.sub(r"\s+", " ", cell.get_text(" ", strip=True)).strip() for cell in row.select("th, td")]
            cells = [cell for cell in cells if cell and not looks_like_boilerplate(cell)]
            if cells:
                row_texts.append(" | ".join(cells))
        if row_texts:
            table_blocks.append("\n".join(row_texts))
    return clean_extracted_text("\n\n".join(table_blocks))


def merge_page_text(headings: list[str], body_text: str, table_text: str) -> str:
    parts = headings + ([body_text] if body_text else []) + ([table_text] if table_text else [])
    return clean_extracted_text("\n".join(part for part in parts if part))


def detect_blocked_page(html: str, title: str, headings: list[str], content: str) -> str | None:
    combined = " ".join(
        part for part in [title, " ".join(headings), content, html[:8000]] if part
    ).lower()

    for pattern in BLOCKED_PAGE_PATTERNS:
        if re.search(pattern, combined):
            return f"Blocked/interstitial page detected via pattern: {pattern}"

    token_count = len(content.split())
    heading_count = len(headings)
    if token_count < 80 and heading_count <= 1:
        if any(
            token in combined
            for token in ["javascript", "cookies", "security", "challenge", "verify", "access denied"]
        ):
            return "Blocked/interstitial page detected from sparse content and challenge language"

    return None


def extract_page_data(url: str, include_internal_links: bool = False, internal_limit: int = 0) -> PageExtraction:
    response = request_url(url)
    html = response.text
    soup = BeautifulSoup(html, "html.parser")

    title = re.sub(r"\s+", " ", soup.title.get_text(" ", strip=True)).strip() if soup.title else ""
    headings = extract_headings_from_soup(soup)
    table_text = extract_relevant_table_text(soup)

    extracted = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=True,
        include_links=False,
        include_images=False,
        favor_precision=True,
        deduplicate=True,
    )
    if extracted:
        content = merge_page_text(headings, extracted, table_text)
    else:
        content = merge_page_text(headings, extract_article_like_text(soup), table_text)

    blocked_reason = detect_blocked_page(html, title, headings, content)
    if blocked_reason:
        raise ValueError(blocked_reason)

    internal_links = []
    if include_internal_links and internal_limit > 0:
        internal_links = extract_internal_links(url, soup, internal_limit)

    return PageExtraction(
        url=normalize_url(url),
        title=title,
        headings=headings,
        content=content,
        internal_links=internal_links,
    )


def safe_extract_page_data(
    url: str,
    include_internal_links: bool = False,
    internal_limit: int = 0,
) -> tuple[PageExtraction | None, str | None]:
    try:
        return extract_page_data(url, include_internal_links=include_internal_links, internal_limit=internal_limit), None
    except RequestException as exc:
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        if status_code:
            return None, f"HTTP {status_code} while fetching {url}"
        return None, f"Request failed for {url}: {exc}"
    except Exception as exc:
        return None, f"Extraction failed for {url}: {exc}"


def fetch_serpapi_results(keyword: str, country: str, limit: int, api_key: str) -> list[RankingRecord]:
    for attempt in range(3):
        try:
            response = requests.get(
                "https://serpapi.com/search.json",
                params={
                    "engine": "google",
                    "q": keyword,
                    "hl": "en",
                    "gl": country,
                    "num": limit,
                    "api_key": api_key,
                },
                timeout=45,
            )
            response.raise_for_status()
            data = response.json()
            break
        except RequestException as exc:
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
            else:
                raise exc

    organic_results = data.get("organic_results", [])
    records: list[RankingRecord] = []
    for index, result in enumerate(organic_results[:limit], start=1):
        link = result.get("link")
        if not link:
            continue
        records.append(
            RankingRecord(
                keyword=keyword,
                rank=index,
                url=normalize_url(link),
                title=result.get("title", ""),
            )
        )
    return records


def get_openai_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)


def extract_entities_with_openai(client: OpenAI, text: str, url: str) -> list[dict]:
    prompt = f"""
Extract important named entities and SEO-relevant concepts from the visible page copy below.

Rules:
- Return only entities or concepts that appear explicitly in the text.
- Focus on items useful for SEO/content analysis.
- Include people, companies, products, locations, technologies, organizations, brands, publications, industry concepts, policy concepts, and topical phrases.
- Normalize obvious duplicates to one canonical label.
- Ignore navigation, cookie banners, and boilerplate.
- Return valid JSON matching this schema:
  {{
    "entities": [
      {{"name": "OpenAI", "type": "organization", "confidence": 0.92}}
    ]
  }}

URL: {url}

CONTENT:
\"\"\"
{text}
\"\"\"
""".strip()

    response = client.responses.create(
        model=ENTITY_EXTRACTION_MODEL,
        input=prompt,
        text={"format": {"type": "json_object"}},
    )
    parsed = json.loads(response.output_text)
    return parsed.get("entities", [])


def find_supporting_snippet(entity: str, source_urls: list[str], content_by_url: dict[str, str]) -> str:
    pattern = re.compile(re.escape(entity), re.IGNORECASE)
    for url in source_urls:
        content = re.sub(r"\s+", " ", content_by_url.get(url, "")).strip()
        match = pattern.search(content)
        if not match:
            continue
        sentence_boundaries = list(re.finditer(r"(?<=[.!?])\s+", content))
        start = 0
        end = len(content)
        for boundary in sentence_boundaries:
            if boundary.end() <= match.start():
                start = boundary.end()
            elif boundary.start() >= match.end():
                end = boundary.start()
                break
        snippet = content[start:end].strip()
        if len(snippet) < 120 and end < len(content):
            next_boundary = next((b.start() for b in sentence_boundaries if b.start() > end), len(content))
            snippet = f"{snippet} {content[end:next_boundary].strip()}".strip()
        return snippet
    return ""


def build_ranking_tables(records: list[RankingRecord]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rankings_df = pd.DataFrame([record.__dict__ for record in records])
    common_urls = (
        rankings_df.groupby("url", as_index=False)
        .agg(
            keyword_count=("keyword", "nunique"),
            avg_rank=("rank", "mean"),
            keywords=("keyword", lambda values: ", ".join(sorted(set(values)))),
            title=("title", "first"),
        )
        .sort_values(by=["keyword_count", "avg_rank"], ascending=[False, True])
    )
    return rankings_df, common_urls


def exclude_client_domain_rows(common_urls_df: pd.DataFrame, client_urls: list[str]) -> pd.DataFrame:
    if common_urls_df.empty or not client_urls:
        return common_urls_df
    client_domains = {extract_domain(url) for url in client_urls}
    return common_urls_df[
        ~common_urls_df["url"].apply(lambda url: extract_domain(url) in client_domains)
    ].copy()


def cluster_entities(
    entities_by_url: dict[str, list[dict]],
    content_by_url: dict[str, str],
    headings_by_url: dict[str, list[str]],
) -> list[dict]:
    grouped: dict[str, dict] = {}
    for url, entities in entities_by_url.items():
        page_keys_seen = set()
        content = content_by_url.get(url, "")
        headings = " ".join(headings_by_url.get(url, []))
        opening = " ".join(content.splitlines()[:4]).lower()
        faq_like = "faq" in content.lower() or "frequently asked" in content.lower()
        for entity in entities:
            label = str(entity.get("name", "")).strip()
            if not label:
                continue
            topic_key = normalize_topic_key(label)
            if not topic_key or topic_key in page_keys_seen:
                continue
            page_keys_seen.add(topic_key)
            row = grouped.setdefault(
                topic_key,
                {
                    "topic_key": topic_key,
                    "entity": label,
                    "entity_type": entity.get("type", "unknown"),
                    "top_url_count": 0,
                    "source_urls": [],
                    "domains": set(),
                    "heading_mentions": 0,
                    "opening_mentions": 0,
                    "faq_mentions": 0,
                    "avg_model_confidence": 0.0,
                    "confidence_samples": [],
                },
            )
            if len(label) > len(row["entity"]):
                row["entity"] = label
            row["top_url_count"] += 1
            row["source_urls"].append(url)
            row["domains"].add(extract_domain(url))
            row["confidence_samples"].append(float(entity.get("confidence", 0) or 0))
            if re.search(re.escape(label), headings, re.IGNORECASE):
                row["heading_mentions"] += 1
            if re.search(re.escape(label), opening, re.IGNORECASE):
                row["opening_mentions"] += 1
            if faq_like and re.search(re.escape(label), content, re.IGNORECASE):
                row["faq_mentions"] += 1

    rows = []
    for row in grouped.values():
        avg_model_conf = sum(row["confidence_samples"]) / max(1, len(row["confidence_samples"]))
        competitor_prominence = min(
            100.0,
            row["top_url_count"] * 12
            + row["heading_mentions"] * 10
            + row["opening_mentions"] * 8
            + row["faq_mentions"] * 5
            + avg_model_conf * 20,
        )
        confidence_label = classify_recommendation_confidence(competitor_prominence)
        evidence = find_supporting_snippet(row["entity"], row["source_urls"], content_by_url)
        competitor_only = (
            row["entity_type"] in {"organization", "product"} and any(
                brand in row["entity"].lower() for brand in COMPETITOR_BRAND_HINTS
            )
        )
        rows.append(
            {
                "entity": row["entity"],
                "entity_type": row["entity_type"],
                "canonical_topic": row["topic_key"],
                "top_url_count": row["top_url_count"],
                "top_domains": ", ".join(sorted(row["domains"])),
                "heading_mentions": row["heading_mentions"],
                "opening_mentions": row["opening_mentions"],
                "faq_mentions": row["faq_mentions"],
                "avg_model_confidence": round(avg_model_conf, 2),
                "competitor_prominence": round(competitor_prominence, 1),
                "confidence_label": confidence_label,
                "competitor_only": competitor_only,
                "top_urls": ", ".join(row["source_urls"]),
                "evidence_snippet": evidence,
                "why_it_matters": (
                    f"Appears on {row['top_url_count']} overlapping ranking URLs; "
                    f"heading mentions: {row['heading_mentions']}; opening mentions: {row['opening_mentions']}."
                ),
            }
        )
    return sorted(rows, key=lambda item: (item["competitor_prominence"], item["top_url_count"]), reverse=True)


def build_competitor_heading_table(headings_by_url: dict[str, list[str]]) -> pd.DataFrame:
    counter = Counter()
    sources = defaultdict(set)
    for url, headings in headings_by_url.items():
        for heading in dedupe_preserve_order(headings):
            key = heading.lower()
            counter[key] += 1
            sources[key].add(extract_domain(url))
    rows = [
        {
            "heading": heading,
            "top_url_count": count,
            "top_domains": ", ".join(sorted(sources[heading.lower()])),
        }
        for heading, count in sorted(
            ((next(orig for orig in counter if orig == key), count) for key, count in counter.items()),
            key=lambda item: item[1],
            reverse=True,
        )
    ]
    if not rows:
        return pd.DataFrame(columns=["heading", "top_url_count", "top_domains"])
    # restore original case from first seen heading
    restored = []
    first_seen = {}
    for url, headings in headings_by_url.items():
        for heading in headings:
            first_seen.setdefault(heading.lower(), heading)
    for key, count in counter.most_common():
        restored.append(
            {
                "heading": first_seen.get(key, key),
                "top_url_count": count,
                "top_domains": ", ".join(sorted(sources[key])),
            }
        )
    return pd.DataFrame(restored)


def build_gap_table(
    clustered_entities: list[dict],
    client_entities_by_url: dict[str, list[dict]],
    client_pages: dict[str, PageExtraction],
) -> pd.DataFrame:
    client_entity_keys = {
        normalize_topic_key(str(entity.get("name", "")).strip())
        for entities in client_entities_by_url.values()
        for entity in entities
        if str(entity.get("name", "")).strip()
    }
    heading_corpus = " ".join(
        heading.lower() for page in client_pages.values() for heading in page.headings
    )
    title_corpus = " ".join(page.title.lower() for page in client_pages.values())
    body_corpus = " ".join(page.content.lower() for page in client_pages.values())

    rows = []
    for item in clustered_entities:
        topic_key = item["canonical_topic"]
        variants = phrase_variants(item["entity"], topic_key)
        literal_body_match = literal_topic_match(body_corpus, variants)
        fuzzy_body_match = token_set_overlap_match(body_corpus, variants)
        literal_heading_match = literal_topic_match(heading_corpus, variants)
        literal_title_match = literal_topic_match(title_corpus, variants)
        body_mention_count = count_topic_mentions(body_corpus, variants)

        machine_detectability = "detected" if topic_key in client_entity_keys else "not detected"
        if machine_detectability == "not detected" and (literal_body_match or fuzzy_body_match):
            machine_detectability = "weakly detected"

        structural_hit = literal_heading_match or literal_title_match
        if machine_detectability == "detected" and structural_hit:
            human_coverage = "already covered"
        elif machine_detectability in {"detected", "weakly detected"} or structural_hit or literal_body_match or fuzzy_body_match:
            human_coverage = "partially covered"
        else:
            human_coverage = "missing"

        if human_coverage == "already covered" and machine_detectability == "detected":
            coverage_status = "clearly covered"
        elif human_coverage in {"already covered", "partially covered"} and machine_detectability in {"weakly detected", "not detected"}:
            coverage_status = "weak signal"
        elif human_coverage == "partially covered":
            coverage_status = "partially covered"
        elif human_coverage == "missing":
            coverage_status = "missing"
        else:
            coverage_status = "unknown"

        client_detectability_score = 100 if machine_detectability == "detected" else 45 if machine_detectability == "weakly detected" else 0
        section_priority = round(item["competitor_prominence"] * 0.65 + (100 - client_detectability_score) * 0.35, 1)
        rows.append(
            {
                **item,
                "machine_detectability": machine_detectability,
                "human_coverage": human_coverage,
                "coverage_status": coverage_status,
                "literal_body_match": literal_body_match,
                "literal_heading_match": literal_heading_match,
                "literal_title_match": literal_title_match,
                "body_mention_count": body_mention_count,
                "client_detectability_score": client_detectability_score,
                "section_priority": section_priority,
                "gap": coverage_status in {"missing", "weak signal", "partially covered"},
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        by=["section_priority", "competitor_prominence", "top_url_count"],
        ascending=[False, False, False],
    )


def describe_first_mention(text: str, term: str) -> str:
    if not text or not term:
        return "not found"
    normalized_text = text.lower()
    normalized_term = term.lower()
    idx = normalized_text.find(normalized_term)
    if idx == -1:
        return "not found"
    ratio = idx / max(len(normalized_text), 1)
    if ratio <= 0.2:
        return "early on page"
    if ratio <= 0.6:
        return "mid-page"
    return "late on page"


def build_section_signal_context(
    sections: list[dict],
    gap_df: pd.DataFrame,
    client_pages: dict[str, PageExtraction],
    client_entities_by_url: dict[str, list[dict]],
) -> list[dict]:
    gap_lookup = {}
    if not gap_df.empty:
        gap_lookup = {str(row["entity"]).lower(): row for _, row in gap_df.iterrows()}

    contexts = []
    for section in sections:
        evidence_rows = []
        for entity in section.get("supporting_entities", []):
            lookup = gap_lookup.get(str(entity).lower())
            client_title_hits = 0
            client_heading_hits = 0
            client_body_mentions = 0
            first_mentions = []
            client_entity_detected = False

            for url, page in client_pages.items():
                entity_detected = any(
                    str(item.get("name", "")).strip().lower() == str(entity).lower()
                    for item in client_entities_by_url.get(url, [])
                )
                if entity_detected:
                    client_entity_detected = True
                if str(entity).lower() in page.title.lower():
                    client_title_hits += 1
                heading_hits = sum(1 for heading in page.headings if str(entity).lower() in heading.lower())
                body_mentions = page.content.lower().count(str(entity).lower())
                client_heading_hits += heading_hits
                client_body_mentions += body_mentions
                first_mentions.append(describe_first_mention(page.content, entity))

            evidence_rows.append(
                {
                    "entity": entity,
                    "competitor_top_url_count": int(lookup["top_url_count"]) if lookup is not None else 0,
                    "competitor_heading_mentions": int(lookup["heading_mentions"]) if lookup is not None else 0,
                    "competitor_opening_mentions": int(lookup["opening_mentions"]) if lookup is not None else 0,
                    "competitor_prominence": float(lookup["competitor_prominence"]) if lookup is not None else 0,
                    "client_entity_detected": client_entity_detected,
                    "client_title_hits": client_title_hits,
                    "client_heading_hits": client_heading_hits,
                    "client_body_mentions": client_body_mentions,
                    "client_first_mention_positions": first_mentions,
                    "literal_body_match": bool(lookup["literal_body_match"]) if lookup is not None else False,
                    "literal_heading_match": bool(lookup["literal_heading_match"]) if lookup is not None else False,
                    "literal_title_match": bool(lookup["literal_title_match"]) if lookup is not None else False,
                    "body_mention_count": int(lookup["body_mention_count"]) if lookup is not None else 0,
                }
            )
        contexts.append({"section": section.get("section", ""), "signal_context": evidence_rows})
    return contexts


def generate_section_recommendations(
    client: OpenAI,
    keywords: list[str],
    intent_summary: dict,
    common_urls_df: pd.DataFrame,
    heading_patterns_df: pd.DataFrame,
    gap_rows_df: pd.DataFrame,
) -> dict:
    common_url_rows = common_urls_df.head(8)[["url", "title", "keyword_count", "avg_rank"]].to_dict("records")
    heading_rows = heading_patterns_df.head(15).to_dict("records")
    gap_rows = gap_rows_df.head(30).to_dict("records")
    prompt = f"""
You are generating section recommendations based on entity analysis and recurring competitor heading patterns.

Return JSON with this schema:
{{
  "executive_summary": "short paragraph",
  "must_cover_sections": [
    {{
      "section": "Section name",
      "why_it_matters": "short explanation",
      "reasoning": "why this section is recommended from the evidence",
      "supporting_entities": ["entity 1", "entity 2"],
      "recommended_topics": ["topic 1", "topic 2"],
      "section_priority": 0,
      "confidence_label": "high | medium | low"
    }}
  ],
  "key_points_to_cover": ["Clear explanation of insurance cover levels and benefits"],
  "trust_signals": ["signal 1", "signal 2"],
  "competitor_only_mentions": ["brand or product 1", "brand or product 2"],
  "section_methodology_note": "one sentence",
  "brief_takeaways": ["takeaway 1", "takeaway 2"]
}}

Rules:
- Base recommendations on repeated entities/concepts, competitor prominence, gaps, and recurring headings.
- Prioritize reusable sections, not competitor brands or products.
- Avoid generic "what is x" sections unless the evidence clearly supports guide-style informational intent.
- Every section must cite 2-5 supporting_entities from the provided gap rows.
- section_priority should broadly align to the provided section_priority signals.
- confidence_label should reflect the underlying evidence strength, not optimism.
- key_points_to_cover should be short recommendation-style statements, not questions.
- Be concise and practical.

KEYWORDS:
{json.dumps(keywords)}

INTENT SUMMARY:
{json.dumps(intent_summary)}

TOP RANKING URL SUMMARY:
{json.dumps(common_url_rows)}

RECURRING COMPETITOR HEADINGS:
{json.dumps(heading_rows)}

ENTITY AND GAP SIGNALS:
{json.dumps(gap_rows)}
""".strip()
    response = client.responses.create(
        model=SYNTHESIS_MODEL,
        input=prompt,
        text={"format": {"type": "json_object"}},
    )
    return json.loads(response.output_text)


def assess_section_coverage(
    client: OpenAI,
    sections: list[dict],
    client_pages: dict[str, PageExtraction],
    client_entities_by_url: dict[str, list[dict]],
    gap_df: pd.DataFrame,
) -> list[dict]:
    if not sections or not client_pages:
        return sections

    client_page_data = []
    for url, page in client_pages.items():
        client_page_data.append(
            {
                "url": url,
                "title": page.title,
                "headings": page.headings,
                "entities": client_entities_by_url.get(url, []),
            }
        )
    signal_context = build_section_signal_context(sections, gap_df, client_pages, client_entities_by_url)

    prompt = f"""
Assess each recommended section against the supplied client pages.

Return JSON:
{{
  "sections": [
    {{
      "section": "Section name",
      "human_coverage": "already covered | partially covered | missing",
      "human_coverage_reasoning": "short explanation",
      "machine_detectability": "detected | weakly detected | not detected",
      "machine_detectability_reasoning": "short explanation"
    }}
  ]
}}

Rules:
- human_coverage should use titles, headings, and entities.
- machine_detectability should reflect whether the concept is clearly represented in extracted entities, not just obvious to a human.
- When a section is weak or partial, explain it comparatively: for example repeated in competitor headings/opening copy, but only briefly mentioned, mentioned late, or not surfaced in headings on the client page.
- Do not imply that a topic must appear in a title or heading to count as covered. Treat titles/headings as stronger signals, not as a requirement.
- If body copy mentions exist, acknowledge that and explain that the issue is lack of prominence or clarity rather than absence.
- Avoid absolute wording like "no coverage" unless there is no meaningful title, heading, body-copy, or entity evidence at all.
- Only say "no mentions" when literal_body_match, literal_heading_match, literal_title_match, client_body_mentions, and client_entity_detected all indicate absence.
- Prefer concrete explanations over generic wording.
- Keep section names unchanged.
- Be concise.

SECTIONS:
{json.dumps(sections)}

CLIENT PAGE DATA:
{json.dumps(client_page_data)}

SECTION SIGNAL CONTEXT:
{json.dumps(signal_context)}
""".strip()
    response = client.responses.create(
        model=SYNTHESIS_MODEL,
        input=prompt,
        text={"format": {"type": "json_object"}},
    )
    parsed = json.loads(response.output_text)
    coverage_by_section = {row.get("section"): row for row in parsed.get("sections", [])}

    enriched = []
    for section in sections:
        coverage = coverage_by_section.get(section.get("section"), {})
        merged = dict(section)
        merged["human_coverage"] = coverage.get("human_coverage", "unknown")
        merged["human_coverage_reasoning"] = coverage.get("human_coverage_reasoning", "")
        merged["machine_detectability"] = coverage.get("machine_detectability", "unknown")
        merged["machine_detectability_reasoning"] = coverage.get("machine_detectability_reasoning", "")
        human_coverage = merged["human_coverage"].lower()
        machine_detectability = merged["machine_detectability"].lower()
        if human_coverage == "already covered" and machine_detectability == "detected":
            merged["coverage_status"] = "clearly covered"
            merged["coverage_reasoning"] = merged["human_coverage_reasoning"]
        elif human_coverage in {"already covered", "partially covered"} and machine_detectability in {"weakly detected", "not detected"}:
            merged["coverage_status"] = "weak signal"
            merged["coverage_reasoning"] = merged["machine_detectability_reasoning"] or merged["human_coverage_reasoning"]
        elif human_coverage == "partially covered":
            merged["coverage_status"] = "partially covered"
            merged["coverage_reasoning"] = merged["human_coverage_reasoning"]
        elif human_coverage == "missing":
            merged["coverage_status"] = "missing"
            merged["coverage_reasoning"] = merged["human_coverage_reasoning"] or merged["machine_detectability_reasoning"]
        else:
            merged["coverage_status"] = "unknown"
            merged["coverage_reasoning"] = merged["human_coverage_reasoning"] or merged["machine_detectability_reasoning"]
        enriched.append(merged)
    return enriched


def render_status_badge(prefix: str, status: str, reasoning: str, styles: dict[str, dict]) -> None:
    style = styles.get(status.lower(), styles["unknown"])
    reasoning_html = f"<div style='margin-top:6px; font-size:0.92rem;'>{reasoning}</div>" if reasoning else ""
    st.markdown(
        f"""
        <div style="
            background:{style['background']};
            border-left:6px solid {style['border']};
            color:{style['text']};
            padding:12px 14px;
            border-radius:10px;
            margin:8px 0 10px 0;
        ">
            <div style="font-weight:700;">{prefix}: {style['label']}</div>
            {reasoning_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def expand_client_urls(seed_urls: list[str], crawl_related: bool) -> tuple[list[str], list[dict]]:
    if not crawl_related:
        return seed_urls, []
    expanded = list(seed_urls)
    notes = []
    for url in seed_urls:
        page, error = safe_extract_page_data(url, include_internal_links=True, internal_limit=MAX_INTERNAL_CLIENT_PAGES)
        if error or not page:
            notes.append({"url": url, "reason": error or "Could not inspect internal links"})
            continue
        for internal in page.internal_links:
            if internal not in expanded:
                expanded.append(internal)
    return dedupe_preserve_order(expanded), notes


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buffer = StringIO()
    df.to_csv(buffer, index=False, quoting=csv.QUOTE_ALL, escapechar="\\")
    return buffer.getvalue().encode("utf-8")


def build_background_export(
    filtered_entities_df: pd.DataFrame,
    competitor_heading_patterns_df: pd.DataFrame,
    competitor_common_urls_df: pd.DataFrame,
    common_urls_df: pd.DataFrame,
    rankings_df: pd.DataFrame,
) -> pd.DataFrame:
    frames = []
    if not filtered_entities_df.empty:
        entity_export = filtered_entities_df.copy()
        entity_export.insert(0, "export_section", "entities")
        frames.append(entity_export)
    if not competitor_heading_patterns_df.empty:
        headings_export = competitor_heading_patterns_df.copy()
        headings_export.insert(0, "export_section", "competitor_headings")
        frames.append(headings_export)
    if not competitor_common_urls_df.empty:
        competitor_urls_export = competitor_common_urls_df.copy()
        competitor_urls_export.insert(0, "export_section", "competitor_common_urls")
        frames.append(competitor_urls_export)
    if not common_urls_df.empty:
        common_urls_export = common_urls_df.copy()
        common_urls_export.insert(0, "export_section", "all_common_urls")
        frames.append(common_urls_export)
    if not rankings_df.empty:
        rankings_export = rankings_df.copy()
        rankings_export.insert(0, "export_section", "all_ranking_rows")
        frames.append(rankings_export)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def build_recommendations_export(section_data: dict, filtered_gap_df: pd.DataFrame) -> pd.DataFrame:
    recommendation_rows = []
    executive_summary = section_data.get("executive_summary", "")
    for section in section_data.get("must_cover_sections", []):
        recommendation_rows.append(
            {
                "export_section": "recommended_sections",
                "section": section.get("section", ""),
                "why_it_matters": section.get("why_it_matters", ""),
                "reasoning": section.get("reasoning", ""),
                "supporting_entities": ", ".join(section.get("supporting_entities", [])),
                "recommended_topics": ", ".join(section.get("recommended_topics", [])),
                "coverage_status": section.get("coverage_status", ""),
                "coverage_reasoning": section.get("coverage_reasoning", ""),
                "confidence_label": section.get("confidence_label", ""),
                "section_priority": section.get("section_priority", ""),
                "executive_summary": executive_summary,
            }
        )

    if recommendation_rows:
        return pd.DataFrame(recommendation_rows)
    return pd.DataFrame()


def run_analysis(
    keywords: list[str],
    country: str,
    results_per_keyword: int,
    entity_limit: int,
    client_urls: list[str],
    serpapi_key: str,
    openai_api_key: str,
    crawl_related_client_pages: bool,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> dict:
    def report_progress(completed: int, total: int, message: str) -> None:
        if progress_callback:
            progress_callback(completed, total, message)

    estimated_total_steps = (
        len(keywords)
        + entity_limit
        + max(1, len(client_urls)) * (2 if crawl_related_client_pages else 1)
        + max(1, len(client_urls))
        + 3
    )
    completed_steps = 0

    ranking_records: list[RankingRecord] = []
    skipped_keywords = []
    for keyword in keywords:
        try:
            ranking_records.extend(
                fetch_serpapi_results(
                    keyword=keyword,
                    country=country,
                    limit=results_per_keyword,
                    api_key=serpapi_key,
                )
            )
        except RequestException as exc:
            skipped_keywords.append({"keyword": keyword, "reason": f"SERP request failed: {exc}"})
        completed_steps += 1
        report_progress(completed_steps, estimated_total_steps, f"Fetched SERP data for {keyword}")

    if not ranking_records:
        raise RuntimeError("No ranking results could be fetched for the supplied keywords.")

    rankings_df, common_urls_df = build_ranking_tables(ranking_records)
    competitor_common_urls_df = exclude_client_domain_rows(common_urls_df, client_urls)
    competitor_target_urls = competitor_common_urls_df.head(entity_limit)["url"].tolist()
    estimated_total_steps = max(
        estimated_total_steps,
        len(keywords) + len(competitor_target_urls) + max(1, len(client_urls)) * (2 if crawl_related_client_pages else 1) + max(1, len(client_urls)) + 3,
    )

    openai_client = get_openai_client(openai_api_key)

    competitor_pages: dict[str, PageExtraction] = {}
    competitor_entities_by_url: dict[str, list[dict]] = {}
    skipped_urls: list[dict] = []
    for url in competitor_target_urls:
        page, error = safe_extract_page_data(url)
        if error or not page:
            skipped_urls.append({"url": url, "reason": error or "Unknown extraction failure", "source": "top_ranking"})
            continue
        competitor_pages[url] = page
        competitor_entities_by_url[url] = extract_entities_with_openai(openai_client, page.content, url)
        completed_steps += 1
        report_progress(completed_steps, estimated_total_steps, f"Analyzed competitor page: {extract_domain(url)}")

    expanded_client_urls, client_crawl_notes = expand_client_urls(client_urls, crawl_related_client_pages)
    skipped_urls.extend({**note, "source": "client_discovery"} for note in client_crawl_notes)
    completed_steps += 1
    report_progress(completed_steps, estimated_total_steps, "Expanded client URLs")

    client_pages: dict[str, PageExtraction] = {}
    client_entities_by_url: dict[str, list[dict]] = {}
    for url in expanded_client_urls:
        page, error = safe_extract_page_data(url)
        if error or not page:
            skipped_urls.append({"url": url, "reason": error or "Unknown extraction failure", "source": "client"})
            continue
        client_pages[url] = page
        client_entities_by_url[url] = extract_entities_with_openai(openai_client, page.content, url)
        completed_steps += 1
        report_progress(completed_steps, estimated_total_steps, f"Analyzed client page: {extract_domain(url)}")

    competitor_content_by_url = {url: page.content for url, page in competitor_pages.items()}
    competitor_headings_by_url = {url: page.headings for url, page in competitor_pages.items()}
    competitor_heading_patterns_df = build_competitor_heading_table(competitor_headings_by_url)
    clustered_entities = cluster_entities(
        competitor_entities_by_url,
        competitor_content_by_url,
        competitor_headings_by_url,
    )
    top_entities_df = pd.DataFrame(clustered_entities)
    gap_df = build_gap_table(clustered_entities, client_entities_by_url, client_pages)
    intent_summary = detect_query_intent(keywords, competitor_common_urls_df)
    completed_steps += 1
    report_progress(completed_steps, estimated_total_steps, "Built topic clusters and gap scoring")

    section_data = generate_section_recommendations(
        openai_client,
        keywords,
        intent_summary,
        competitor_common_urls_df,
        competitor_heading_patterns_df,
        gap_df,
    )
    completed_steps += 1
    report_progress(completed_steps, estimated_total_steps, "Generated section recommendations")
    section_data["must_cover_sections"] = assess_section_coverage(
        openai_client,
        section_data.get("must_cover_sections", []),
        client_pages,
        client_entities_by_url,
        gap_df,
    )
    completed_steps += 1
    report_progress(completed_steps, estimated_total_steps, "Assessed client coverage and machine detectability")

    return {
        "rankings_df": rankings_df,
        "common_urls_df": common_urls_df,
        "competitor_common_urls_df": competitor_common_urls_df,
        "competitor_heading_patterns_df": competitor_heading_patterns_df,
        "top_entities_df": top_entities_df,
        "gap_df": gap_df,
        "section_data": section_data,
        "intent_summary": intent_summary,
        "competitor_pages": competitor_pages,
        "competitor_entities_by_url": competitor_entities_by_url,
        "client_pages": client_pages,
        "client_entities_by_url": client_entities_by_url,
        "skipped_urls": skipped_urls,
        "skipped_keywords": skipped_keywords,
    }


st.set_page_config(page_title="Content Gap Mapper", layout="wide")
st.markdown(APP_CSS, unsafe_allow_html=True)
st.markdown(
    """
    <section class="hero-shell">
        <h1 class="hero-title">Content Gap Mapper</h1>
        <p class="hero-copy">
            Add a keyword cluster, compare top-ranking pages with client content, and surface the strongest
            content gaps and section recommendations without wading through raw analysis first.
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)

with st.expander("What this tool does, why it is useful, and its limitations"):
    st.markdown(
        """
**What it does**

- Takes a cluster of keywords and pulls the top-ranking URLs for each one.
- Finds the ranking pages that appear most often across that cluster.
- Extracts visible page copy and headings from those pages.
- Identifies recurring entities and topical phrases.
- Compares those patterns against the client URL or URLs you provide.
- Surfaces the main topics that may need stronger coverage and suggests what to include.

**Why it is useful**

- Helps you quickly spot the themes competitors emphasise most.
- Shows where client content may be lighter, weaker, or less explicit.
- Turns a large amount of SERP and page analysis into a simpler recommendation view.
- Gives supporting evidence so recommendations are not just black-box suggestions.

**Limitations**

- It is based on extracted page content, so blocked or messy pages can reduce accuracy.
- Recommendations are inferred from recurring patterns, not copied directly from competitor page structure.
- A topic can be present on the page but still register as weak if it is not strongly signalled in the extracted copy.
- The output is directional and should support editorial judgement, not replace it.
"""
    )

with st.sidebar:
    st.header("Inputs")
    serpapi_key = get_secret("SERPAPI_API_KEY")
    openai_api_key = get_secret("OPENAI_API_KEY")
    country = st.text_input(
        "Country code",
        value=DEFAULT_COUNTRY,
        help="Country used for the Google results pull, for example `uk` or `us`.",
    )
    crawl_related_client_pages = st.toggle(
        "Check nearby client pages",
        value=False,
        help="Also check a small number of internal pages linked from the supplied client URLs to get a broader view of topic coverage.",
    )

    st.header("Entity filters")
    hide_competitor_brands = st.toggle(
        "Hide competitor brands/products",
        value=True,
        help="Removes branded competitor names and branded products so the tables focus more on reusable topics.",
    )
    hide_locations = st.toggle(
        "Hide locations",
        value=False,
        help="Removes geographic entities like countries or regions from the tables.",
    )
    recommendation_mode = st.selectbox(
        "Recommendation focus",
        options=["Gaps only", "Gaps and partial coverage", "Show all"],
        index=2,
        help="Controls whether recommendations show only missing sections, missing plus partial coverage, or every suggested section.",
    )

keyword_text = st.text_area(
    f"Cluster terms (up to {MAX_KEYWORDS})",
    height=180,
    placeholder="best running shoes\nrunning shoes for flat feet\ntrail running shoes",
)
client_url_text = st.text_area(
    "Client URLs (optional, one per line)",
    height=120,
    placeholder="https://client.com/running-shoes\nhttps://client.com/trail-shoes",
)

keywords = dedupe_preserve_order([line.strip() for line in keyword_text.splitlines() if line.strip()])
client_urls = dedupe_preserve_order([normalize_url(line.strip()) for line in client_url_text.splitlines() if line.strip()])

run_clicked = st.button("Run analysis", type="primary", use_container_width=True)

if run_clicked:
    if not keywords:
        st.error("Add at least one keyword.")
    elif len(keywords) > MAX_KEYWORDS:
        st.error(f"Add no more than {MAX_KEYWORDS} keywords.")
    elif not serpapi_key:
        st.error("Add a SerpAPI key.")
    elif not openai_api_key:
        st.error("Add an OpenAI API key.")
    else:
        progress_bar = st.progress(0.0, text="Starting analysis...")
        progress_message = st.empty()
        eta_message = st.empty()
        started_at = time.time()

        def update_progress(completed: int, total: int, message: str) -> None:
            ratio = min(1.0, completed / max(total, 1))
            elapsed = max(time.time() - started_at, 0.01)
            remaining_steps = max(total - completed, 0)
            seconds_per_step = elapsed / max(completed, 1)
            eta_seconds = int(remaining_steps * seconds_per_step)
            progress_bar.progress(ratio, text=message)
            progress_message.caption(f"Progress: {completed}/{total} steps")
            eta_message.caption(f"Estimated time left: ~{eta_seconds}s")

        try:
                results = run_analysis(
                    keywords=keywords,
                    country=country.strip().lower() or DEFAULT_COUNTRY,
                    results_per_keyword=DEFAULT_RESULTS_PER_KEYWORD,
                    entity_limit=10,
                    client_urls=client_urls,
                    serpapi_key=serpapi_key,
                    openai_api_key=openai_api_key,
                    crawl_related_client_pages=crawl_related_client_pages,
                    progress_callback=update_progress,
            )
        except Exception as exc:
            progress_bar.empty()
            progress_message.empty()
            eta_message.empty()
            st.exception(exc)
        else:
            progress_bar.progress(1.0, text="Analysis complete")
            progress_message.caption("Progress: complete")
            eta_message.caption(f"Completed in ~{int(time.time() - started_at)}s")
            rankings_df = results["rankings_df"]
            common_urls_df = results["common_urls_df"]
            competitor_common_urls_df = results["competitor_common_urls_df"]
            competitor_heading_patterns_df = results["competitor_heading_patterns_df"]
            top_entities_df = results["top_entities_df"]
            gap_df = results["gap_df"]
            section_data = results["section_data"]
            intent_summary = results["intent_summary"]
            competitor_pages = results["competitor_pages"]
            competitor_entities_by_url = results["competitor_entities_by_url"]
            client_pages = results["client_pages"]
            client_entities_by_url = results["client_entities_by_url"]
            skipped_urls = results["skipped_urls"]
            skipped_keywords = results["skipped_keywords"]

            filtered_entities_df = top_entities_df.copy()
            filtered_gap_df = gap_df.copy()
            if not filtered_entities_df.empty:
                filtered_entities_df = filtered_entities_df[
                    filtered_entities_df["competitor_prominence"] >= DEFAULT_MIN_PROMINENCE
                ]
                if hide_competitor_brands:
                    filtered_entities_df = filtered_entities_df[~filtered_entities_df["competitor_only"]]
                if hide_locations:
                    filtered_entities_df = filtered_entities_df[
                        filtered_entities_df["entity_type"].str.lower() != "location"
                    ]
            if not filtered_gap_df.empty:
                filtered_gap_df = filtered_gap_df[
                    filtered_gap_df["competitor_prominence"] >= DEFAULT_MIN_PROMINENCE
                ]
                if hide_competitor_brands:
                    filtered_gap_df = filtered_gap_df[~filtered_gap_df["competitor_only"]]
                if hide_locations:
                    filtered_gap_df = filtered_gap_df[
                        filtered_gap_df["entity_type"].str.lower() != "location"
                    ]

            sections = section_data.get("must_cover_sections", [])
            if recommendation_mode == "Gaps only":
                sections = [s for s in sections if s.get("coverage_status", "").lower() == "missing"]
            elif recommendation_mode == "Gaps and partial coverage":
                sections = [
                    s for s in sections if s.get("coverage_status", "").lower() in {"missing", "partially covered", "weak signal"}
                ]
            section_data["must_cover_sections"] = sections

            if skipped_keywords:
                st.warning(f"Skipped {len(skipped_keywords)} keyword SERP request(s).")
                st.dataframe(pd.DataFrame(skipped_keywords), use_container_width=True)
            if skipped_urls:
                st.warning(f"Skipped {len(skipped_urls)} URL(s) due to fetch or extraction issues.")
                st.dataframe(pd.DataFrame(skipped_urls), use_container_width=True)

            methodology_note = section_data.get(
                "section_methodology_note",
                "Recommendations are inferred from recurring entity and heading patterns across overlapping ranking URLs.",
            )
            background_export_df = build_background_export(
                filtered_entities_df,
                competitor_heading_patterns_df,
                competitor_common_urls_df,
                common_urls_df,
                rankings_df,
            )
            recommendations_export_df = build_recommendations_export(section_data, filtered_gap_df)

            rec_tab, gaps_tab, urls_tab, exports_tab, debug_tab = st.tabs(
                ["Section recommendations", "Gap tables", "Ranking URLs", "Exports", "Debug"]
            )

            with rec_tab:
                with st.expander("Methodology and limitations"):
                    st.markdown(
                        """
**Methodology**

- Collect top-ranking URLs across the keyword cluster.
- Identify the URLs that recur most often across those SERPs.
- Extract visible headings and body copy from those recurring URLs.
- Extract recurring entities and concepts from that content.
- Cluster near-duplicate concepts into canonical topics.
- Score each topic by competitor prominence using repeated URL presence, heading mentions, opening-copy mentions, and extraction confidence.
- Compare those topics against the supplied client URLs and, optionally, related internal client pages.
- Generate section recommendations from the combined entity, heading, intent, and gap signals.

**Limitations**

- Recommendations are inferred from machine-readable patterns, not copied directly from competitor page structure.
- Entity extraction can miss broad concepts even when a human would say the page clearly covers them.
- Scraping failures or noisy HTML can reduce coverage quality.
- Related client-page crawling is shallow and domain-limited.
- Confidence labels indicate evidence strength, not guaranteed SEO impact.
"""
                    )

                col1, col2, col3 = st.columns(3)
                col1.metric("Primary intent", intent_summary.get("primary_intent", "unknown").title())
                col2.metric("Competitor URLs analyzed", len(competitor_pages))
                col3.metric("Client pages analyzed", len(client_pages))

                if methodology_note:
                    st.info(methodology_note)

                with st.expander("How to read these recommendations"):
                    st.markdown(
                        """
- **Coverage status** combines whether the page appears to cover the topic and how clearly that topic comes through in the extracted text.
- **Clearly covered** means the topic appears to be well covered and comes through strongly.
- **Partially covered** means the topic is present, but not strongly or completely enough yet.
- **Weak signal** means the topic may be present, but the page does not signal it clearly.
- **Gap** means the topic does not appear clearly represented.
- Use **Priority** and **Confidence** to judge which sections are most worth reviewing first.
"""
                    )

                st.subheader("Executive summary")
                st.write(section_data.get("executive_summary", ""))

                section_rows = section_data.get("must_cover_sections", [])
                if section_rows:
                    missing_count = sum(
                        1 for section in section_rows if str(section.get("coverage_status", "")).lower() == "missing"
                    )
                    partial_count = sum(
                        1
                        for section in section_rows
                        if str(section.get("coverage_status", "")).lower() in {"partially covered", "weak signal"}
                    )
                    strong_signal_count = sum(
                        1
                        for section in section_rows
                        if str(section.get("coverage_status", "")).lower() == "clearly covered"
                    )
                    metric1, metric2, metric3 = st.columns(3)
                    metric1.metric("Topics needing stronger coverage", missing_count)
                    metric2.metric("Partly covered topics", partial_count)
                    metric3.metric("Clearly covered topics", strong_signal_count)

                st.subheader("Recommended sections")
                if not section_data.get("must_cover_sections"):
                    st.write("No sections match the current recommendation filter.")
                for section in section_data.get("must_cover_sections", []):
                    with st.container(border=True):
                        st.markdown(f"**{section.get('section', 'Untitled section')}**")
                        color = CONFIDENCE_STYLES.get(section.get("confidence_label", "low"), "#334155")
                        st.markdown(
                            f"<div style='color:{color}; font-weight:700;'>Confidence: {section.get('confidence_label', 'unknown').title()} | Priority: {section.get('section_priority', 'n/a')}</div>",
                            unsafe_allow_html=True,
                        )
                        st.write(section.get("why_it_matters", ""))
                        if section.get("reasoning"):
                            st.caption(f"Why this is recommended: {section['reasoning']}")
                        render_status_badge(
                            "Coverage status",
                            section.get("coverage_status", "unknown"),
                            section.get("coverage_reasoning", ""),
                            STATUS_STYLES,
                        )
                        if section.get("supporting_entities"):
                            st.caption("Supported by entities: " + ", ".join(section["supporting_entities"]))
                        if section.get("recommended_topics"):
                            st.write("Include:")
                            st.write(", ".join(section["recommended_topics"]))

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Key points to cover")
                    for item in section_data.get("key_points_to_cover", section_data.get("important_questions", [])):
                        st.write(f"- {item}")
                with col2:
                    st.subheader("Trust signals to include")
                    for item in section_data.get("trust_signals", []):
                        st.write(f"- {item}")

                st.subheader("Key takeaways")
                for item in section_data.get("brief_takeaways", []):
                    st.write(f"- {item}")

                competitor_only = section_data.get("competitor_only_mentions", [])
                if competitor_only:
                    st.subheader("Competitor-only mentions to avoid overweighting")
                    st.write(", ".join(competitor_only))

                with st.expander("Top topic gaps snapshot"):
                    if filtered_gap_df.empty:
                        st.write("No gap data available for the current filters.")
                    else:
                        snapshot_cols = [
                            "entity",
                            "section_priority",
                            "coverage_status",
                            "competitor_prominence",
                        ]
                        st.dataframe(
                            filtered_gap_df[snapshot_cols].head(12),
                            use_container_width=True,
                        )

            with gaps_tab:
                st.info(
                    "This tab is the detailed analysis layer. Use it when you want to audit the underlying entity and gap signals behind the recommendations."
                )

                with st.expander("Recurring entities across common URLs", expanded=False):
                    st.dataframe(filtered_entities_df, use_container_width=True)

                if client_urls:
                    with st.expander("Client gap view", expanded=False):
                        st.dataframe(filtered_gap_df, use_container_width=True)

                with st.expander("Recurring competitor headings", expanded=False):
                    st.dataframe(competitor_heading_patterns_df, use_container_width=True)

            with urls_tab:
                st.info("Use this tab when you want to inspect the actual ranking URL set behind the analysis.")
                if client_urls:
                    with st.expander("Common ranking URLs excluding client domains", expanded=False):
                        st.dataframe(competitor_common_urls_df, use_container_width=True)
                with st.expander("All common ranking URLs", expanded=False):
                    st.dataframe(common_urls_df, use_container_width=True)
                with st.expander("All ranking rows", expanded=False):
                    st.dataframe(rankings_df, use_container_width=True)

            with exports_tab:
                st.subheader("Downloads")
                st.download_button(
                    "Download background data (CSV)",
                    dataframe_to_csv_bytes(background_export_df),
                    file_name="background_data.csv",
                    mime="text/csv",
                    help="Background analysis export including filtered entities, competitor heading patterns, common URLs, and ranking rows.",
                )
                st.download_button(
                    "Download gaps and recommendations (CSV)",
                    dataframe_to_csv_bytes(recommendations_export_df),
                    file_name="gaps_and_recommendations.csv",
                    mime="text/csv",
                    help="Recommendations export including recommended sections and the filtered topic gap view.",
                )

            with debug_tab:
                st.subheader("Analyzed competitor text")
                for url, page in competitor_pages.items():
                    with st.expander(url):
                        st.write(f"Title: {page.title}")
                        st.write("Headings:")
                        st.json(page.headings)
                        st.text_area(f"Analyzed text: {url}", value=page.content, height=220, key=f"comp-{url}")
                        st.write("Entities:")
                        st.json(competitor_entities_by_url.get(url, []))

                if client_pages:
                    st.subheader("Analyzed client text")
                    for url, page in client_pages.items():
                        with st.expander(url):
                            st.write(f"Title: {page.title}")
                            st.write("Headings:")
                            st.json(page.headings)
                            st.text_area(f"Client analyzed text: {url}", value=page.content, height=220, key=f"client-{url}")
                            st.write("Entities:")
                            st.json(client_entities_by_url.get(url, []))

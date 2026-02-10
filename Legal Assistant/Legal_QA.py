import os
import json
import re
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Optional OpenAI (works if OPENAI_API_KEY is set)
try:
    from openai import OpenAI
    _openai_client = OpenAI()
except Exception:
    _openai_client = None


# =========================
# CONFIG (SIMPLE + ROBUST)
# =========================
# Priority order:
# 1) LAW_DATA_PATH env (exact file)
# 2) LAW_DATA_DIR env (directory)
# 3) ./data directory in project
# 4) fallback: try current working dir (json scan)

CANDIDATE_FILENAMES = [
    "labour_law_chunks_multilingual_cleaned.json",
    "labour_law_articles_multilingual_cleaned.json",
    "labour_law_articles_ar_cleaned.json",
    "labour_law_chunks.json",
    "labour_law_articles.json",
]


# =========================
# ARABIC NORMALIZATION
# =========================
_AR_DIACRITICS = re.compile(r"[\u064B-\u0652\u0670\u0640]")  # tashkeel + tatweel


def _normalize_arabic(text: str) -> str:
    """
    Normalize Arabic for retrieval robustness:
    - remove diacritics/tatweel
    - normalize alef variants -> ا
    - ة -> ه (helps "الخدمة" vs "الخدمه")
    - ى -> ي
    - remove punctuation
    """
    if not text:
        return ""

    t = str(text)
    t = _AR_DIACRITICS.sub("", t)

    # Normalize Alef variants
    t = t.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")

    # Normalize taa marbuta + alif maqsoora
    t = t.replace("ة", "ه").replace("ى", "ي")

    # Normalize hamza on waw/yaa
    t = t.replace("ؤ", "و").replace("ئ", "ي")

    # Remove punctuation/symbols but keep letters/numbers/spaces
    t = re.sub(r"[^\w\s\u0600-\u06FF]", " ", t, flags=re.UNICODE)

    # Collapse whitespace
    t = re.sub(r"\s+", " ", t).strip().lower()
    return t


def _normalize_english(text: str) -> str:
    if not text:
        return ""
    t = str(text).lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _is_arabic_text(s: str) -> bool:
    return bool(re.search(r"[\u0600-\u06FF]", s or ""))


def _normalize_query(q: str) -> str:
    # If query contains Arabic, apply Arabic normalization; otherwise English normalization.
    if _is_arabic_text(q):
        return _normalize_arabic(q)
    return _normalize_english(q)


def _tokenize(text: str) -> List[str]:
    t = _normalize_query(text)
    if not t:
        return []
    # token split
    parts = re.split(r"\s+", t)
    # keep short tokens only if not too tiny
    return [p for p in parts if p and len(p) >= 2]


# =========================
# DATASET DISCOVERY / LOAD
# =========================
def _search_in_dir(directory: Path) -> Optional[Path]:
    if not directory.exists() or not directory.is_dir():
        return None

    # Try known filenames first
    for name in CANDIDATE_FILENAMES:
        p = directory / name
        if p.exists() and p.is_file():
            return p

    # If not found, scan JSON files in folder (prefer multilingual+chunks)
    jsons = [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == ".json"]
    if not jsons:
        return None

    preferred = None
    for p in jsons:
        low = p.name.lower()
        if "multilingual" in low and "chunks" in low:
            preferred = p
            break
    return preferred or jsons[0]


def _find_dataset_file(data_path: Optional[str] = None, data_dir: Optional[str] = None) -> str:
    # 0) explicit passed file
    if data_path:
        p = Path(data_path).expanduser().resolve()
        if p.exists() and p.is_file():
            return str(p)

    # 1) env: exact file path
    env_path = os.getenv("LAW_DATA_PATH")
    if env_path:
        p = Path(env_path).expanduser().resolve()
        if p.exists() and p.is_file():
            return str(p)

    # 2) explicit passed directory
    if data_dir:
        d = Path(data_dir).expanduser().resolve()
        found = _search_in_dir(d)
        if found:
            return str(found)

    # 3) env: directory
    env_dir = os.getenv("LAW_DATA_DIR")
    if env_dir:
        d = Path(env_dir).expanduser().resolve()
        found = _search_in_dir(d)
        if found:
            return str(found)

    # 4) local ./data in project folder (near this file)
    here = Path(__file__).parent
    local_data = here / "data"
    found = _search_in_dir(local_data)
    if found:
        return str(found)

    # 5) local ./data in CWD
    cwd_data = Path(os.getcwd()) / "data"
    found = _search_in_dir(cwd_data)
    if found:
        return str(found)

    # 6) scan cwd jsons (last resort)
    cwd = Path(os.getcwd())
    found = _search_in_dir(cwd)
    if found:
        return str(found)

    raise FileNotFoundError(
        "Could not find labour law dataset.\n"
        "Tried:\n"
        "- passed data_path / data_dir\n"
        "- LAW_DATA_PATH env\n"
        "- LAW_DATA_DIR env\n"
        "- ./data (near Legal_QA.py)\n"
        "- ./data (cwd)\n"
        "- scanning json in cwd\n"
        "Expected filenames:\n" + "\n".join([f"- {x}" for x in CANDIDATE_FILENAMES])
    )


def _safe_load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # dict with list inside
    if isinstance(data, dict):
        for key in ["articles", "chunks", "data", "items"]:
            if key in data and isinstance(data[key], list):
                return data[key]
        return [data]

    if isinstance(data, list):
        return data

    return []


def _normalize_records(raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalize records to:
    {
      "article_id": int|str,
      "text": str,
      "meta": dict
    }
    """
    norm: List[Dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue

        article_id = (
            item.get("article_id")
            or item.get("article_number")
            or item.get("id")
            or item.get("article")
            or item.get("number")
        )

        text = (
            item.get("text")
            or item.get("content")
            or item.get("chunk")
            or item.get("body")
            or item.get("article_text")
        )

        if text is None and isinstance(item.get("data"), dict):
            text = item["data"].get("text") or item["data"].get("content")

        if not text:
            continue

        # make numeric ids usable
        article_id_val: Any = article_id
        try:
            if isinstance(article_id, str) and article_id.strip().isdigit():
                article_id_val = int(article_id.strip())
            elif isinstance(article_id, (int, float)):
                article_id_val = int(article_id)
        except Exception:
            article_id_val = article_id

        norm.append({
            "article_id": article_id_val,
            "text": str(text).strip(),
            "meta": {k: v for k, v in item.items() if k not in ["text", "content", "chunk", "body", "article_text"]},
        })
    return norm


# =========================
# ARTICLE QUERY PARSER
# =========================
def _parse_article_query(q: str) -> Optional[Tuple[int, int]]:
    """
    Detect:
    - "article 5"
    - "articles 1-10"
    - "المادة 5"
    - "المواد 1-10"
    Returns (start, end) inclusive
    """
    q0 = (q or "").strip().lower()

    # english range
    m = re.search(r"(articles?|art\.?)\s*(\d+)\s*[-–to]+\s*(\d+)", q0)
    if m:
        a = int(m.group(2)); b = int(m.group(3))
        return (min(a, b), max(a, b))

    # english single
    m = re.search(r"(articles?|art\.?)\s*(\d+)", q0)
    if m:
        a = int(m.group(2))
        return (a, a)

    # arabic range
    m = re.search(r"(المواد|المادة)\s*(\d+)\s*[-–إلىto]+\s*(\d+)", q0)
    if m:
        a = int(m.group(2)); b = int(m.group(3))
        return (min(a, b), max(a, b))

    # arabic single
    m = re.search(r"(المواد|المادة)\s*(\d+)", q0)
    if m:
        a = int(m.group(2))
        return (a, a)

    return None


# =========================
# BM25 (SIMPLE, NO EXTRA LIBS)
# =========================
class _BM25:
    def __init__(self, docs_tokens: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs_tokens = docs_tokens
        self.doc_count = len(docs_tokens)
        self.doc_lens = [len(toks) for toks in docs_tokens]
        self.avgdl = sum(self.doc_lens) / self.doc_count if self.doc_count else 0.0

        # df + idf
        self.df: Dict[str, int] = {}
        for toks in docs_tokens:
            for term in set(toks):
                self.df[term] = self.df.get(term, 0) + 1

        self.idf: Dict[str, float] = {}
        for term, df in self.df.items():
            # BM25+ style idf (stable)
            self.idf[term] = math.log(1 + (self.doc_count - df + 0.5) / (df + 0.5))

        # term frequencies per doc
        self.tf: List[Dict[str, int]] = []
        for toks in docs_tokens:
            d: Dict[str, int] = {}
            for t in toks:
                d[t] = d.get(t, 0) + 1
            self.tf.append(d)

    def score(self, query_tokens: List[str], idx: int) -> float:
        if idx < 0 or idx >= self.doc_count:
            return 0.0
        tf = self.tf[idx]
        dl = self.doc_lens[idx] or 1
        score = 0.0
        for term in query_tokens:
            if term not in tf:
                continue
            freq = tf[term]
            idf = self.idf.get(term, 0.0)
            denom = freq + self.k1 * (1 - self.b + self.b * (dl / (self.avgdl or 1.0)))
            score += idf * ((freq * (self.k1 + 1)) / (denom or 1.0))
        return score


# =========================
# DATA CACHE (SAFE FOR STREAMLIT)
# =========================
_DATA_CACHE: Dict[str, Any] = {
    "path": None,
    "articles": None,
    "bm25": None,
    "tokens": None,
}


def _load_data_cached(data_path: Optional[str] = None, data_dir: Optional[str] = None) -> Tuple[str, List[Dict[str, Any]], _BM25]:
    resolved = _find_dataset_file(data_path=data_path, data_dir=data_dir)

    if _DATA_CACHE["path"] == resolved and _DATA_CACHE["articles"] is not None and _DATA_CACHE["bm25"] is not None:
        return resolved, _DATA_CACHE["articles"], _DATA_CACHE["bm25"]

    raw = _safe_load_json(resolved)
    articles = _normalize_records(raw)

    # Build tokens once
    docs_tokens: List[List[str]] = []
    for a in articles:
        docs_tokens.append(_tokenize(a["text"]))

    bm25 = _BM25(docs_tokens)

    _DATA_CACHE["path"] = resolved
    _DATA_CACHE["articles"] = articles
    _DATA_CACHE["tokens"] = docs_tokens
    _DATA_CACHE["bm25"] = bm25
    return resolved, articles, bm25


# =========================
# RETRIEVAL
# =========================
def retrieve_articles(query: str, top_k: int = 3, data_path: Optional[str] = None, data_dir: Optional[str] = None) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Deterministic retrieval:
    1) If query asks for article number/range -> return those (as many as needed).
    2) Else BM25 ranking (robust for Arabic/English).
    """
    dataset_path, articles, bm25 = _load_data_cached(data_path=data_path, data_dir=data_dir)
    if not articles:
        return dataset_path, []

    # Explicit article requests
    ar = _parse_article_query(query)
    if ar:
        start, end = ar
        selected = [a for a in articles if isinstance(a["article_id"], int) and start <= a["article_id"] <= end]
        if selected:
            # return full range if user asked range; otherwise top_k at least 1
            if start != end:
                return dataset_path, selected
            return dataset_path, selected[:max(1, top_k)]

    # BM25 for general queries
    q_tokens = _tokenize(query)
    if not q_tokens:
        return dataset_path, articles[:top_k]

    scored: List[Tuple[float, Dict[str, Any]]] = []
    for i, a in enumerate(articles):
        s = bm25.score(q_tokens, i)
        if s > 0:
            scored.append((s, a))

    scored.sort(key=lambda x: x[0], reverse=True)
    best = [a for _, a in scored][:top_k]

    # If BM25 returns nothing (rare), fallback to first k
    if not best:
        best = articles[:top_k]

    return dataset_path, best


def _build_context(retrieved: List[Dict[str, Any]]) -> str:
    parts = []
    for a in retrieved:
        parts.append(f"Article {a['article_id']}:\n{a['text']}")
    return "\n\n".join(parts)


# =========================
# MAIN CHATBOT
# =========================
def legal_qa_chatbot(
    query: str,
    top_k: int = 3,
    return_metadata: bool = False,
    data_path: Optional[str] = None,
    data_dir: Optional[str] = None
) -> Any:
    dataset_path, retrieved = retrieve_articles(query, top_k=top_k, data_path=data_path, data_dir=data_dir)
    context = _build_context(retrieved)
    retrieved_ids = []
    for a in retrieved:
        try:
            # keep ints for evaluation
            if isinstance(a["article_id"], int):
                retrieved_ids.append(a["article_id"])
            elif isinstance(a["article_id"], str) and a["article_id"].strip().isdigit():
                retrieved_ids.append(int(a["article_id"].strip()))
            else:
                retrieved_ids.append(a["article_id"])
        except Exception:
            retrieved_ids.append(a.get("article_id"))

    # If no API key / no OpenAI client, return extractive
    if _openai_client is None or not os.getenv("OPENAI_API_KEY"):
        answer = (
            "No API key found. Returning retrieved legal text only.\n\n"
            + (context[:5000] if context else "No context retrieved.")
        )
        if return_metadata:
            return {
                "answer": answer,
                "retrieved_article_ids": retrieved_ids,
                "context": context,
                "dataset_path": dataset_path,
            }
        return answer

    arabic = _is_arabic_text(query)

    system_prompt = (
        "You are a Jordanian Legal Assistant.\n"
        "Strict rules:\n"
        "1) Answer ONLY using the provided legal text (context). Do not add outside facts.\n"
        "2) If the answer is not found in the context, say exactly:\n"
        "   - Arabic: 'غير موجود في النص القانوني المرفق.'\n"
        "   - English: 'Not found in the provided legal text.'\n"
        "3) Do NOT provide legal advice. Provide informational summary only.\n"
        "4) Answer in Arabic if the user asked in Arabic; otherwise answer in English.\n"
        "5) When you mention a rule, cite the article number(s) you used.\n"
        "6) If multiple articles are retrieved, keep the answer short and grounded.\n"
    )

    user_prompt = (
        f"Legal Text (Context):\n{context}\n\n"
        f"Question:\n{query}\n\n"
        "Answer:"
    )

    resp = _openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )
    answer = (resp.choices[0].message.content or "").strip()

    if return_metadata:
        return {
            "answer": answer,
            "retrieved_article_ids": retrieved_ids,
            "context": context,
            "dataset_path": dataset_path,
        }
    return answer

import os
import json
import re
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Generator

import numpy as np

# Optional OpenAI
try:
    from openai import OpenAI
    from openai.types.chat import ChatCompletionMessageParam
    _openai_client = OpenAI()
    _OPENAI_AVAILABLE = True
except Exception:
    _openai_client = None
    _OPENAI_AVAILABLE = False
    ChatCompletionMessageParam = dict  # type: ignore

# Optional: embeddings module for hybrid retrieval
try:
    from embeddings import compute_and_cache_embeddings, embed_query, cosine_similarity_scores
    _EMBEDDINGS_MODULE = True
except ImportError:
    _EMBEDDINGS_MODULE = False

# =========================
# CONFIG
# =========================
CANDIDATE_FILENAMES = [
    "labour_law_chunks_multilingual_cleaned.json",
    "labour_law_articles_multilingual_cleaned.json",
    "labour_law_articles_ar_cleaned.json",
    "labour_law_chunks.json",
    "labour_law_articles.json",
]

MAX_CHAT_HISTORY = 20  # Keep last 20 messages for context

# Minimum BM25 score threshold — below this, warn user about low confidence
CONFIDENCE_THRESHOLD = 0.5

# =========================
# ARABIC NORMALIZATION
# =========================
_AR_DIACRITICS = re.compile(r"[\u064B-\u0652\u0670\u0640]")


def _normalize_arabic(text: str) -> str:
    if not text:
        return ""
    t = str(text)
    t = _AR_DIACRITICS.sub("", t)
    t = t.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    t = t.replace("ة", "ه").replace("ى", "ي")
    t = t.replace("ؤ", "و").replace("ئ", "ي")
    t = re.sub(r"[^\w\s\u0600-\u06FF]", " ", t, flags=re.UNICODE)
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
    arabic_chars = sum(1 for c in (s or "") if "\u0600" <= c <= "\u06FF")
    total_chars = len([c for c in (s or "") if c.strip()])
    if total_chars == 0:
        return False
    # For mixed queries: if ANY Arabic characters present, treat as Arabic
    # This handles cases like "What is المادة 32?" correctly
    if arabic_chars > 0:
        return True
    return (arabic_chars / total_chars) > 0.3


def _normalize_query(q: str) -> str:
    if _is_arabic_text(q):
        return _normalize_arabic(q)
    return _normalize_english(q)


def _tokenize(text: str) -> List[str]:
    t = _normalize_query(text)
    if not t:
        return []
    parts = re.split(r"\s+", t)
    return [p for p in parts if p and len(p) >= 2]


# =========================
# DATASET DISCOVERY
# =========================
def _search_in_dir(directory: Path) -> Optional[Path]:
    if not directory.exists() or not directory.is_dir():
        return None
    for name in CANDIDATE_FILENAMES:
        p = directory / name
        if p.exists() and p.is_file():
            return p
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
    if data_path:
        p = Path(data_path).expanduser().resolve()
        if p.exists() and p.is_file():
            return str(p)
    env_path = os.getenv("LAW_DATA_PATH")
    if env_path:
        p = Path(env_path).expanduser().resolve()
        if p.exists() and p.is_file():
            return str(p)
    if data_dir:
        d = Path(data_dir).expanduser().resolve()
        found = _search_in_dir(d)
        if found:
            return str(found)
    env_dir = os.getenv("LAW_DATA_DIR")
    if env_dir:
        d = Path(env_dir).expanduser().resolve()
        found = _search_in_dir(d)
        if found:
            return str(found)
    here = Path(__file__).parent
    local_data = here / "data"
    found = _search_in_dir(local_data)
    if found:
        return str(found)
    cwd_data = Path(os.getcwd()) / "data"
    found = _search_in_dir(cwd_data)
    if found:
        return str(found)
    cwd = Path(os.getcwd())
    found = _search_in_dir(cwd)
    if found:
        return str(found)
    raise FileNotFoundError(
        "Could not find labour law dataset.\n"
        "Expected filenames:\n" + "\n".join([f"- {x}" for x in CANDIDATE_FILENAMES])
    )


def _safe_load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        for key in ["articles", "chunks", "data", "items"]:
            if key in data and isinstance(data[key], list):
                return data[key]
        return [data]
    if isinstance(data, list):
        return data
    return []


def _normalize_records(raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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
        if not text:
            continue

        article_id_val: Any = article_id
        try:
            if isinstance(article_id, str) and article_id.strip().isdigit():
                article_id_val = int(article_id.strip())
            elif isinstance(article_id, (int, float)):
                article_id_val = int(article_id)
        except Exception:
            article_id_val = article_id

        # Preserve lang field if present
        lang = item.get("lang", None)
        # Auto-detect lang if not present
        if lang is None:
            lang = "ar" if _is_arabic_text(str(text)) else "en"

        norm.append({
            "article_id": article_id_val,
            "text": str(text).strip(),
            "lang": lang,
            "meta": {k: v for k, v in item.items() if k not in ["text", "content", "chunk", "body", "article_text"]},
        })
    return norm


# =========================
# ARTICLE QUERY PARSER
# =========================
def _parse_article_query(q: str) -> Optional[Tuple[int, int]]:
    q0 = (q or "").strip().lower()
    m = re.search(r"(articles?|art\.?)\s*(\d+)\s*[-–to]+\s*(\d+)", q0)
    if m:
        a = int(m.group(2)); b = int(m.group(3))
        return (min(a, b), max(a, b))
    m = re.search(r"(articles?|art\.?)\s*(\d+)", q0)
    if m:
        a = int(m.group(2))
        return (a, a)
    m = re.search(r"(المواد|المادة)\s*(\d+)\s*[-–إلىto]+\s*(\d+)", q0)
    if m:
        a = int(m.group(2)); b = int(m.group(3))
        return (min(a, b), max(a, b))
    m = re.search(r"(المواد|المادة)\s*(\d+)", q0)
    if m:
        a = int(m.group(2))
        return (a, a)
    return None


# =========================
# BM25
# =========================
class _BM25:
    def __init__(self, docs_tokens: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs_tokens = docs_tokens
        self.doc_count = len(docs_tokens)
        self.doc_lens = [len(toks) for toks in docs_tokens]
        self.avgdl = sum(self.doc_lens) / self.doc_count if self.doc_count else 0.0
        self.df: Dict[str, int] = {}
        for toks in docs_tokens:
            for term in set(toks):
                self.df[term] = self.df.get(term, 0) + 1
        self.idf: Dict[str, float] = {}
        for term, df in self.df.items():
            self.idf[term] = math.log(1 + (self.doc_count - df + 0.5) / (df + 0.5))
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
# DATA CACHE (TWO: EN + AR)
# =========================
_DATA_CACHE: Dict[str, Any] = {
    "path": None,
    "articles_en": None,
    "articles_ar": None,
    "bm25_en": None,
    "bm25_ar": None,
    # Embeddings: numpy array or None (not yet computed) or "failed"
    "embeddings_en": None,
    "embeddings_ar": None,
}


def _load_data_cached(data_path: Optional[str] = None, data_dir: Optional[str] = None):
    resolved = _find_dataset_file(data_path=data_path, data_dir=data_dir)

    if _DATA_CACHE["path"] == resolved and _DATA_CACHE["articles_en"] is not None:
        return resolved, _DATA_CACHE["articles_en"], _DATA_CACHE["articles_ar"], _DATA_CACHE["bm25_en"], _DATA_CACHE["bm25_ar"]

    raw = _safe_load_json(resolved)
    all_articles = _normalize_records(raw)

    # Split by language
    articles_en = [a for a in all_articles if a.get("lang") == "en"]
    articles_ar = [a for a in all_articles if a.get("lang") == "ar"]

    # Fallback: if no lang tags, use all for both
    if not articles_en and not articles_ar:
        articles_en = all_articles
        articles_ar = all_articles

    def build_bm25(articles):
        tokens = [_tokenize(a["text"]) for a in articles]
        return _BM25(tokens)

    bm25_en = build_bm25(articles_en) if articles_en else None
    bm25_ar = build_bm25(articles_ar) if articles_ar else None

    _DATA_CACHE["path"] = resolved
    _DATA_CACHE["articles_en"] = articles_en
    _DATA_CACHE["articles_ar"] = articles_ar
    _DATA_CACHE["bm25_en"] = bm25_en
    _DATA_CACHE["bm25_ar"] = bm25_ar

    return resolved, articles_en, articles_ar, bm25_en, bm25_ar


# =========================
# RETRIEVAL (LANGUAGE-AWARE)
# =========================
def retrieve_articles(
    query: str,
    top_k: int = 3,
    lang_override: Optional[str] = None,
    data_path: Optional[str] = None,
    data_dir: Optional[str] = None,
) -> Tuple[str, List[Dict[str, Any]], float, str]:
    """
    Returns: (dataset_path, articles, confidence_score, detected_lang)
    confidence_score: 0.0-1.0, used for hallucination guard
    """
    dataset_path, articles_en, articles_ar, bm25_en, bm25_ar = _load_data_cached(
        data_path=data_path, data_dir=data_dir
    )

    # Detect language
    detected_lang = lang_override or ("ar" if _is_arabic_text(query) else "en")

    # Pick correct article set and BM25
    if detected_lang == "ar":
        articles = articles_ar or []
        bm25 = bm25_ar
    else:
        articles = articles_en or []
        bm25 = bm25_en

    if not articles:
        return dataset_path, [], 0.0, detected_lang

    # Explicit article number requests → return all matching articles
    ar_range = _parse_article_query(query)
    if ar_range:
        start, end = ar_range
        # Get all chunks for the requested range
        range_chunks = [a for a in articles if isinstance(a["article_id"], int) and start <= a["article_id"] <= end]
        if range_chunks:
            # Deduplicate: keep only the first chunk per article_id (avoids sending duplicate context)
            seen_ids: set = set()
            selected = []
            for a in range_chunks:
                if a["article_id"] not in seen_ids:
                    seen_ids.add(a["article_id"])
                    selected.append(a)
            confidence = 1.0  # Direct article lookup = full confidence
            return dataset_path, selected, confidence, detected_lang

    # ── BM25 retrieval ────────────────────────────────────────────────────────
    q_tokens = _tokenize(query)
    bm25_scores = np.zeros(len(articles), dtype=np.float64)
    if q_tokens and bm25:
        for i in range(len(articles)):
            bm25_scores[i] = bm25.score(q_tokens, i)

    # Normalize BM25 scores to [0, 1]
    bm25_max = bm25_scores.max()
    bm25_norm = bm25_scores / bm25_max if bm25_max > 0 else bm25_scores

    # ── Semantic retrieval (hybrid) ───────────────────────────────────────────
    cosine_norm = np.zeros(len(articles), dtype=np.float64)
    emb_key = f"embeddings_{detected_lang}"

    if _EMBEDDINGS_MODULE and os.getenv("OPENAI_API_KEY"):
        # Lazy-compute article embeddings on first query
        if _DATA_CACHE[emb_key] is None:
            result = compute_and_cache_embeddings(articles, detected_lang)
            _DATA_CACHE[emb_key] = result if result is not None else "failed"

        emb_matrix = _DATA_CACHE[emb_key]
        if isinstance(emb_matrix, np.ndarray):
            q_vec = embed_query(query)
            if q_vec is not None:
                cosine_scores = cosine_similarity_scores(q_vec, emb_matrix)
                # Shift to [0, 1] (cosine can be negative)
                cosine_min = cosine_scores.min()
                cosine_range = cosine_scores.max() - cosine_min
                if cosine_range > 0:
                    cosine_norm = (cosine_scores - cosine_min) / cosine_range

    # ── Hybrid scoring (40% BM25 + 60% semantic) ─────────────────────────────
    alpha = 0.4  # weight for BM25
    hybrid = alpha * bm25_norm + (1.0 - alpha) * cosine_norm

    # If both are zero vectors (no tokens, no embeddings), use first top_k
    if hybrid.max() == 0:
        best = articles[:top_k]
        confidence = 0.1
        return dataset_path, best, confidence, detected_lang

    top_indices = np.argsort(hybrid)[::-1][:top_k]
    best = [articles[i] for i in top_indices]
    top_hybrid_scores = [float(hybrid[i]) for i in top_indices]

    # Confidence from hybrid top score
    confidence = min(top_hybrid_scores[0], 1.0) if top_hybrid_scores else 0.0
    # Boost confidence when BM25 had strong signal (direct keyword match)
    if bm25_max > 5:
        confidence = min(confidence + 0.2, 1.0)

    return dataset_path, best, confidence, detected_lang


def _build_context(retrieved: List[Dict[str, Any]]) -> str:
    parts = []
    for a in retrieved:
        parts.append(f"Article {a['article_id']}:\n{a['text']}")
    return "\n\n".join(parts)


def get_article_by_id(
    article_id: int,
    lang: str = "en",
    data_path: Optional[str] = None,
    data_dir: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Retrieve a single article's full text by its ID.
    Used by the Citation Deep-Dive panel in the UI.

    Returns the normalized article dict or None if not found.
    """
    try:
        _, articles_en, articles_ar, _, _ = _load_data_cached(
            data_path=data_path, data_dir=data_dir
        )
    except Exception:
        return None

    articles = articles_en if lang == "en" else articles_ar
    target = int(str(article_id))
    for a in articles:
        try:
            if int(str(a["article_id"])) == target:
                return a
        except Exception:
            continue
    return None


# =========================
# FOLLOW-UP SUGGESTIONS
# =========================
def _get_followup_suggestions(query: str, lang: str) -> List[str]:
    """Return 3 contextually relevant follow-up questions."""
    suggestions_en = {
        "termination": [
            "What is the required notice period for termination?",
            "What compensation is owed upon termination?",
            "Can an employer terminate without notice?",
        ],
        "salary": [
            "What is the minimum wage in Jordan?",
            "When must salaries be paid?",
            "What deductions are allowed from salary?",
        ],
        "leave": [
            "How many annual leave days am I entitled to?",
            "What are the rules for sick leave?",
            "Is maternity leave paid in Jordan?",
        ],
        "contract": [
            "What must be included in an employment contract?",
            "What is the maximum probation period?",
            "Can a contract be verbal or must it be written?",
        ],
        "overtime": [
            "What is the overtime pay rate?",
            "What are the maximum working hours per week?",
            "Are there restrictions on overtime work?",
        ],
    }
    suggestions_ar = {
        "termination": [
            "ما هي مدة الإشعار المطلوبة عند إنهاء العقد؟",
            "ما هي التعويضات المستحقة عند إنهاء الخدمة؟",
            "هل يمكن للصاحب العمل إنهاء العقد دون إشعار؟",
        ],
        "salary": [
            "ما هو الحد الأدنى للأجور في الأردن؟",
            "متى يجب دفع الرواتب؟",
            "ما هي الاستقطاعات المسموح بها من الراتب؟",
        ],
        "leave": [
            "كم عدد أيام الإجازة السنوية؟",
            "ما هي قواعد الإجازة المرضية؟",
            "هل إجازة الأمومة مدفوعة الأجر في الأردن؟",
        ],
        "contract": [
            "ما الذي يجب تضمينه في عقد العمل؟",
            "ما هي أقصى مدة للتجربة؟",
            "هل يمكن أن يكون العقد شفهياً أم يجب أن يكون مكتوباً؟",
        ],
        "overtime": [
            "ما هو معدل أجر العمل الإضافي؟",
            "ما هو الحد الأقصى لساعات العمل في الأسبوع؟",
            "هل هناك قيود على العمل الإضافي؟",
        ],
    }

    q_lower = query.lower()
    suggestions = suggestions_ar if lang == "ar" else suggestions_en

    if any(w in q_lower for w in ["terminat", "dismiss", "fire", "فصل", "إنهاء", "طرد"]):
        return suggestions["termination"]
    if any(w in q_lower for w in ["salary", "wage", "pay", "راتب", "أجر", "مرتب"]):
        return suggestions["salary"]
    if any(w in q_lower for w in ["leave", "vacation", "holiday", "إجازة", "عطلة"]):
        return suggestions["leave"]
    if any(w in q_lower for w in ["contract", "agreement", "عقد", "اتفاقية"]):
        return suggestions["contract"]
    if any(w in q_lower for w in ["overtime", "hours", "إضافي", "ساعات"]):
        return suggestions["overtime"]

    # Default suggestions
    if lang == "ar":
        return [
            "ما هي حقوق العامل عند إنهاء العقد؟",
            "كم عدد أيام الإجازة السنوية المستحقة؟",
            "ما هو الحد الأدنى للأجور في الأردن؟",
        ]
    return [
        "What are employee rights upon contract termination?",
        "How many annual leave days am I entitled to?",
        "What is the minimum wage in Jordan?",
    ]


# =========================
# MAIN CHATBOT (V2)
# =========================
def legal_qa_chatbot(
    query: str,
    top_k: int = 3,
    return_metadata: bool = False,
    lang_override: Optional[str] = None,
    chat_history: Optional[List[Dict[str, str]]] = None,
    data_path: Optional[str] = None,
    data_dir: Optional[str] = None,
) -> Any:
    """
    V2 Legal QA chatbot.
    - Language-aware retrieval (EN queries → EN chunks, AR queries → AR chunks)
    - Response language always matches query language
    - Hallucination guard (low confidence warning)
    - Follow-up suggestions
    - Chat history support (last 20 messages)
    - lang_override: force 'en' or 'ar' regardless of query language
    """
    dataset_path, retrieved, confidence, detected_lang = retrieve_articles(
        query,
        top_k=top_k,
        lang_override=lang_override,
        data_path=data_path,
        data_dir=data_dir,
    )

    context = _build_context(retrieved)

    retrieved_ids = []
    for a in retrieved:
        try:
            if isinstance(a["article_id"], int):
                retrieved_ids.append(a["article_id"])
            elif isinstance(a["article_id"], str) and a["article_id"].strip().isdigit():
                retrieved_ids.append(int(a["article_id"].strip()))
            else:
                retrieved_ids.append(a["article_id"])
        except Exception:
            retrieved_ids.append(a.get("article_id"))

    low_confidence = confidence < CONFIDENCE_THRESHOLD

    # No API key fallback
    if _openai_client is None or not os.getenv("OPENAI_API_KEY"):
        answer = (
            "⚠️ No API key found. Returning retrieved legal text only.\n\n"
            + (context[:5000] if context else "No context retrieved.")
        )
        if return_metadata:
            return {
                "answer": answer,
                "retrieved_article_ids": retrieved_ids,
                "context": context,
                "dataset_path": dataset_path,
                "confidence": confidence,
                "low_confidence": low_confidence,
                "detected_lang": detected_lang,
                "followup_suggestions": _get_followup_suggestions(query, detected_lang),
            }
        return answer

    # Language-specific system prompt
    if detected_lang == "ar":
        system_prompt = (
            "أنت مساعد قانوني متخصص في قانون العمل الأردني رقم (8) لسنة 1996.\n\n"
            "القواعد الصارمة:\n"
            "1) أجب فقط بناءً على النصوص القانونية المقدمة. لا تضف معلومات خارجية.\n"
            "2) إذا لم تجد الإجابة في النص، قل بوضوح: 'غير موجود في النص القانوني المرفق.'\n"
            "3) اذكر أرقام المواد صراحةً في إجابتك.\n"
            "4) هذا ليس استشارة قانونية - للأغراض المعلوماتية فقط.\n"
            "5) **أجب دائماً باللغة العربية** بغض النظر عن لغة النص المسترجع.\n"
            "6) أضف في نهاية كل إجابة: 'تنويه: هذه المعلومات لأغراض تعليمية فقط ولا تُعدّ استشارة قانونية.'\n"
            "7) إذا طلب المستخدم شيئاً خارج نطاق قانون العمل الأردني (مثل كتابة رسائل، نصائح عامة، قوانين أخرى)، "
            "قل بوضوح: 'هذا خارج نطاق اختصاصي. أنا متخصص فقط في قانون العمل الأردني رقم (8) لسنة 1996.'\n"
        )
    else:
        system_prompt = (
            "You are a legal assistant specialized in Jordanian Labour Law No. 8 of 1996.\n\n"
            "Strict rules:\n"
            "1) Answer ONLY using the provided legal text. Do not add outside information.\n"
            "2) If the answer is not found in the text, say clearly: 'Not found in the provided legal text.'\n"
            "3) Always cite article numbers explicitly in your answer.\n"
            "4) This is NOT legal advice - for informational purposes only.\n"
            "5) **Always answer in English** regardless of the language of the retrieved text.\n"
            "6) End every answer with: 'Disclaimer: This information is for educational purposes only and does not constitute legal advice.'\n"
            "7) If the user asks for anything outside Jordanian Labour Law (e.g. writing letters, general advice, "
            "other laws, math, jokes, or unrelated tasks), respond ONLY with: "
            "'This is outside my scope. I am specialized in Jordanian Labour Law No. 8 of 1996 only.'\n"
        )

    # Build messages with chat history
    messages: List[ChatCompletionMessageParam] = [{"role": "system", "content": system_prompt}]

    # Add recent chat history with token budget to prevent context overflow
    # Each legal answer can be 300-500 words, so we trim by total char count not just count
    if chat_history:
        recent = chat_history[-(MAX_CHAT_HISTORY):]
        # Keep only last N chars of history to stay within context window
        # Approx 12000 chars = ~3000 tokens, safe buffer for gpt-4o-mini (128k context)
        MAX_HISTORY_CHARS = 12000
        total_chars = 0
        trimmed = []
        for msg in reversed(recent):
            msg_chars = len(msg.get("content", ""))
            if total_chars + msg_chars > MAX_HISTORY_CHARS:
                break
            trimmed.insert(0, msg)
            total_chars += msg_chars
        for msg in trimmed:
            messages.append({"role": msg["role"], "content": msg["content"]})  # type: ignore

    # Current query with context
    user_prompt = (
        f"Legal Text (Context):\n{context}\n\n"
        f"User Request: {query}\n\n"
        "Instructions: Respond exactly to what the user asked. "
        "If they asked to summarize, provide a clear summary. "
        "If they asked to explain, explain in simple terms. "
        "If they asked a question, answer it directly. "
        "Always cite the article number(s).\n\n"
        "Response:"
    )
    messages.append({"role": "user", "content": user_prompt})  # type: ignore

    resp = _openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0,
    )
    answer = (resp.choices[0].message.content or "").strip()

    followup = _get_followup_suggestions(query, detected_lang)

    if return_metadata:
        return {
            "answer": answer,
            "retrieved_article_ids": retrieved_ids,
            "context": context,
            "dataset_path": dataset_path,
            "confidence": confidence,
            "low_confidence": low_confidence,
            "detected_lang": detected_lang,
            "followup_suggestions": followup,
        }
    return answer


# =========================
# STREAMING VERSION
# =========================
def legal_qa_chatbot_stream(
    query: str,
    top_k: int = 3,
    lang_override: Optional[str] = None,
    chat_history: Optional[List[Dict[str, str]]] = None,
    data_path: Optional[str] = None,
    data_dir: Optional[str] = None,
) -> Tuple[Generator, Dict[str, Any]]:
    """
    Streaming version. Returns (stream_generator, metadata_dict).
    Use metadata for article IDs, confidence, etc. while streaming the answer.
    """
    dataset_path, retrieved, confidence, detected_lang = retrieve_articles(
        query,
        top_k=top_k,
        lang_override=lang_override,
        data_path=data_path,
        data_dir=data_dir,
    )

    context = _build_context(retrieved)

    retrieved_ids = []
    for a in retrieved:
        try:
            if isinstance(a["article_id"], int):
                retrieved_ids.append(a["article_id"])
            elif isinstance(a["article_id"], str) and a["article_id"].strip().isdigit():
                retrieved_ids.append(int(a["article_id"].strip()))
            else:
                retrieved_ids.append(a["article_id"])
        except Exception:
            retrieved_ids.append(a.get("article_id"))

    low_confidence = confidence < CONFIDENCE_THRESHOLD
    followup = _get_followup_suggestions(query, detected_lang)

    metadata = {
        "retrieved_article_ids": retrieved_ids,
        "context": context,
        "dataset_path": dataset_path,
        "confidence": confidence,
        "low_confidence": low_confidence,
        "detected_lang": detected_lang,
        "followup_suggestions": followup,
    }

    if _openai_client is None or not os.getenv("OPENAI_API_KEY"):
        def no_key_gen():
            yield "⚠️ No API key found. Please set OPENAI_API_KEY."
        return no_key_gen(), metadata

    if detected_lang == "ar":
        system_prompt = (
            "أنت مساعد قانوني متخصص في قانون العمل الأردني رقم (8) لسنة 1996.\n\n"
            "القواعد الصارمة:\n"
            "1) أجب فقط بناءً على النصوص القانونية المقدمة.\n"
            "2) اذكر أرقام المواد صراحةً.\n"
            "3) **أجب دائماً باللغة العربية.**\n"
            "4) أضف تنويهاً في النهاية: 'تنويه: للأغراض التعليمية فقط.'\n"
        )
    else:
        system_prompt = (
            "You are a legal assistant specialized in Jordanian Labour Law No. 8 of 1996.\n\n"
            "Strict rules:\n"
            "1) Answer ONLY using the provided legal text.\n"
            "2) Always cite article numbers.\n"
            "3) **Always answer in English.**\n"
            "4) End with: 'Disclaimer: For educational purposes only.'\n"
        )

    messages: List[ChatCompletionMessageParam] = [{"role": "system", "content": system_prompt}]
    if chat_history:
        recent = chat_history[-(MAX_CHAT_HISTORY):]
        MAX_HISTORY_CHARS = 12000
        total_chars = 0
        trimmed = []
        for msg in reversed(recent):
            msg_chars = len(msg.get("content", ""))
            if total_chars + msg_chars > MAX_HISTORY_CHARS:
                break
            trimmed.insert(0, msg)
            total_chars += msg_chars
        for msg in trimmed:
            messages.append({"role": msg["role"], "content": msg["content"]})  # type: ignore

    user_prompt = (
        f"Legal Text (Context):\n{context}\n\n"
        f"User Request: {query}\n\n"
        "Instructions: Respond exactly to what the user asked. "
        "If they asked to summarize, provide a clear summary. "
        "If they asked to explain, explain in simple terms. "
        "If they asked a question, answer it directly. "
        "Always cite the article number(s).\n\nResponse:"
    )
    messages.append({"role": "user", "content": user_prompt})  # type: ignore

    stream = _openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0,
        stream=True,
    )

    def stream_gen():
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    return stream_gen(), metadata
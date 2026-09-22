"""arXiv 检索适配器：本地 HTTP，对外仍叫 arxiv_search，便于以后换成 MCP server。"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from typing import Any

import httpx

from dl_agent.domain.models import ExternalRef
from dl_agent.mcp_gateway.base import ToolResult, ToolSpec

logger = logging.getLogger(__name__)

ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV_ATOM = "{http://arxiv.org/schemas/atom}"
DEFAULT_ARXIV_API_URL = "https://export.arxiv.org/api/query"
OPENALEX_API_URL = "https://api.openalex.org/works"
NO_ARXIV_HITS = "NO_ARXIV_HITS"
ARXIV_UNAVAILABLE = "ARXIV_UNAVAILABLE"
DEFAULT_MAX_RESULTS = 5
MAX_RESULTS_CAP = 8
SEARCH_POOL_CAP = 8
MAX_QUERY_TERMS = 5
USER_AGENT = "dl-agent/0.1 (arxiv_search; paper reading assistant)"
_FIELD_QUERY = re.compile(r"\b(?:ti|au|abs|cat|all|co|jr):", re.I)
_QUERY_STOP = {
    "arxiv",
    "paper",
    "papers",
    "related",
    "search",
    "look",
    "find",
    "query",
    "literature",
    "pdf",
    "abs",
    "www",
    "http",
    "https",
    "the",
    "and",
    "for",
    "on",
    "in",
    "of",
    "to",
    "with",
    "about",
}
_WITHDRAWN = re.compile(
    r"\bwithdrawn\b|"
    r"\bthis (?:paper|article|submission) has been removed\b|"
    r"\bremoved by arxiv\b|"
    r"\bremoved because\b",
    re.I,
)
_INTENT_PHRASES = re.compile(
    "|".join(
        sorted(
            [
                r"在\s*arxiv\s*上",
                r"arxiv",
                r"检索相关论文",
                r"检索相关文献",
                r"相关论文",
                r"相关文献",
                r"相关工作",
                r"搜一下",
                r"查一下",
                r"找几篇",
                r"找一篇",
                r"帮我",
                r"给我",
                r"看看",
                r"网上",
                r"文献库",
                r"后续工作",
                r"这个方向",
                r"某一方向",
                r"该方向",
                r"search papers",
                r"related papers",
                r"look up",
                r"检索",
                r"搜索",
                r"查找",
                r"论文",
                r"文献",
                r"请",
                r"找",
            ],
            key=len,
            reverse=True,
        )
    ),
    re.IGNORECASE,
)

ARXIV_TOOL_SPEC = ToolSpec(
    name="arxiv_search",
    description=(
        "在 arXiv 检索相关论文元数据（标题、摘要、id）。不是当前这篇 PDF 的原文。"
        "仅当用户问后续工作、相关文献、或本篇证据不够且问题指向外部文献库时使用。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "关键词或标题"},
            "arxiv_id": {"type": "string", "description": "如 2303.08774"},
            "max_results": {"type": "integer", "minimum": 1, "maximum": MAX_RESULTS_CAP},
        },
    },
    source="arxiv",
)


def search_arxiv(
    args: dict[str, Any],
    *,
    client: httpx.Client | None = None,
    timeout: float = 20.0,
    base_url: str = DEFAULT_ARXIV_API_URL,
) -> ToolResult:
    query = str(args.get("query") or "").strip()
    arxiv_id = normalize_arxiv_id(str(args.get("arxiv_id") or args.get("id") or ""))
    limit = _clamp_limit(args.get("max_results"))
    headers = {"User-Agent": USER_AGENT}
    if arxiv_id:
        return _request(
            {"id_list": arxiv_id, "max_results": limit},
            client=client,
            timeout=timeout,
            base_url=base_url,
            headers=headers,
            include_withdrawn=True,
            limit=limit,
        )
    search_query = build_arxiv_search_query(query)
    if not search_query:
        return ToolResult(ok=True, sentinel=NO_ARXIV_HITS)
    arxiv_timeout = timeout if client is not None else min(float(timeout), 8.0)
    result = _request(
        {
            "search_query": search_query,
            "start": 0,
            "max_results": _search_pool(limit),
            "sortBy": "relevance",
        },
        client=client,
        timeout=arxiv_timeout,
        base_url=base_url,
        headers=headers,
        include_withdrawn=False,
        limit=limit,
    )
    if result.sentinel != ARXIV_UNAVAILABLE:
        return result
    if client is not None:
        retry_query = build_arxiv_search_query(_shorten_query(query)) or search_query
        logger.warning("arxiv_search retry query=%s", retry_query)
        return _request(
            {
                "search_query": retry_query,
                "start": 0,
                "max_results": limit,
            },
            client=client,
            timeout=timeout,
            base_url=base_url,
            headers=headers,
            include_withdrawn=False,
            limit=limit,
        )
    logger.warning("arxiv_search falling back to OpenAlex query=%s", query)
    return _search_openalex(query, limit=limit, timeout=timeout)


def _request(
    params: dict[str, str | int],
    *,
    client: httpx.Client | None,
    timeout: float,
    base_url: str,
    headers: dict[str, str],
    include_withdrawn: bool,
    limit: int,
) -> ToolResult:
    timeout_cfg = httpx.Timeout(timeout, connect=min(8.0, float(timeout)))
    try:
        response = _http_get(
            base_url,
            params=params,
            timeout=timeout_cfg,
            headers=headers,
            client=client,
        )
    except (httpx.TimeoutException, httpx.RequestError) as exc:
        logger.warning(
            "arxiv_search transport failed url=%s query=%s err=%s",
            base_url,
            params.get("search_query") or params.get("id_list"),
            exc,
        )
        return ToolResult(ok=False, sentinel=ARXIV_UNAVAILABLE)
    if response.status_code >= 400:
        logger.warning(
            "arxiv_search http %s query=%s body=%s",
            response.status_code,
            params.get("search_query") or params.get("id_list"),
            (response.text or "")[:200],
        )
        return ToolResult(ok=False, sentinel=ARXIV_UNAVAILABLE)
    try:
        refs = parse_atom(response.text, include_withdrawn=include_withdrawn)[:limit]
    except ET.ParseError:
        logger.warning(
            "arxiv_search atom parse failed query=%s body=%s",
            params.get("search_query") or params.get("id_list"),
            (response.text or "")[:200],
        )
        return ToolResult(ok=False, sentinel=ARXIV_UNAVAILABLE)
    if not refs:
        return ToolResult(ok=True, sentinel=NO_ARXIV_HITS)
    return ToolResult(ok=True, text=format_arxiv_hits(refs), refs=refs)


def parse_atom(xml: str, *, include_withdrawn: bool = True) -> list[ExternalRef]:
    root = ET.fromstring(xml)
    out: list[ExternalRef] = []
    for entry in root.findall(f"{ATOM}entry"):
        raw_id = _text(entry.find(f"{ATOM}id"))
        identifier = normalize_arxiv_id(raw_id)
        title = _collapse(_text(entry.find(f"{ATOM}title")))
        if not title:
            continue
        snippet = _collapse(_text(entry.find(f"{ATOM}summary")))
        comment = _collapse(_text(entry.find(f"{ARXIV_ATOM}comment")))
        if not include_withdrawn and is_withdrawn_text(title, snippet, comment):
            continue
        published = _text(entry.find(f"{ATOM}published"))
        year = _year(published)
        authors = [
            _collapse(_text(node.find(f"{ATOM}name")))
            for node in entry.findall(f"{ATOM}author")
        ]
        authors = [name for name in authors if name]
        href = _alternate_url(entry)
        if not href and identifier:
            href = f"https://arxiv.org/abs/{identifier}"
        label = f"arxiv:{identifier}" if identifier else None
        out.append(
            ExternalRef(
                source="arxiv",
                title=title,
                url=href,
                identifier=label,
                snippet=snippet[:800],
                year=year,
                authors=authors,
            )
        )
    return out


def build_arxiv_search_query(query: str) -> str:
    """空格在 arXiv `all:` 里会被拆成 OR；含 arXiv 的问句会命中撤回声明。"""
    text = strip_search_intent(query)
    if not text:
        return ""
    if _FIELD_QUERY.search(text):
        return text
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_.+-]*|[\u4e00-\u9fff]+", text)
    tokens = [token for token in tokens if token.lower() not in _QUERY_STOP]
    if not tokens:
        return ""
    words: list[str] = []
    for token in tokens[:MAX_QUERY_TERMS]:
        safe = token.replace('"', "").replace("-", " ").replace("+", " ").strip()
        words.extend(part for part in safe.split() if part)
    if not words:
        return ""
    phrase = " ".join(words)
    return f'all:"{phrase}"'


def _shorten_query(query: str) -> str:
    text = strip_search_intent(query)
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_.+-]*|[\u4e00-\u9fff]+", text)
    tokens = [token for token in tokens if token.lower() not in _QUERY_STOP]
    return " ".join(tokens[:3])


def strip_search_intent(text: str) -> str:
    leftover = _INTENT_PHRASES.sub(" ", text or "")
    leftover = re.sub(r"[\s，。？！、,.!?;:：；“”‘’（）()【】\[\]《》]+", " ", leftover)
    return leftover.strip()


def is_withdrawn_text(*parts: str) -> bool:
    blob = " ".join(part for part in parts if part)
    return bool(_WITHDRAWN.search(blob))


def parse_openalex(payload: dict[str, Any], limit: int) -> list[ExternalRef]:
    out: list[ExternalRef] = []
    for work in payload.get("results") or []:
        if not isinstance(work, dict):
            continue
        title = _collapse(str(work.get("display_name") or ""))
        if not title:
            continue
        snippet = _openalex_abstract(work.get("abstract_inverted_index"))
        if is_withdrawn_text(title, snippet):
            continue
        ident = _openalex_arxiv_id(work)
        authors = [
            _collapse(str(((item.get("author") or {}).get("display_name")) or ""))
            for item in (work.get("authorships") or [])
            if isinstance(item, dict)
        ]
        authors = [name for name in authors if name]
        year = work.get("publication_year")
        year_int = int(year) if isinstance(year, int) else _year(str(year or ""))
        if ident:
            href = f"https://arxiv.org/abs/{ident}"
            source = "arxiv"
            label = f"arxiv:{ident}"
        else:
            ids = work.get("ids") if isinstance(work.get("ids"), dict) else {}
            loc = work.get("primary_location") if isinstance(work.get("primary_location"), dict) else {}
            href = str(ids.get("doi") or loc.get("landing_page_url") or "") or None
            source = "web"
            label = None
        out.append(
            ExternalRef(
                source=source,
                title=title,
                url=href,
                identifier=label,
                snippet=snippet[:800],
                year=year_int,
                authors=authors,
            )
        )
    arxiv_refs = [item for item in out if item.source == "arxiv"]
    other_refs = [item for item in out if item.source != "arxiv"]
    return (arxiv_refs + other_refs)[:limit]


def _search_openalex(query: str, *, limit: int, timeout: float) -> ToolResult:
    headers = {
        "User-Agent": "dl-agent/0.1 (mailto:paper-writing-assistant@users.noreply.github.com)"
    }
    params = {"search": query, "per_page": max(limit, 8)}
    timeout_cfg = httpx.Timeout(min(float(timeout), 15.0), connect=5.0)
    try:
        with httpx.Client(follow_redirects=True, timeout=timeout_cfg, headers=headers) as session:
            response = session.get(OPENALEX_API_URL, params=params)
    except (httpx.TimeoutException, httpx.RequestError) as exc:
        logger.warning("openalex_search transport failed query=%s err=%s", query, exc)
        return ToolResult(ok=False, sentinel=ARXIV_UNAVAILABLE)
    if response.status_code >= 400:
        logger.warning(
            "openalex_search http %s query=%s body=%s",
            response.status_code,
            query,
            (response.text or "")[:200],
        )
        return ToolResult(ok=False, sentinel=ARXIV_UNAVAILABLE)
    try:
        payload = response.json()
    except ValueError:
        return ToolResult(ok=False, sentinel=ARXIV_UNAVAILABLE)
    if not isinstance(payload, dict):
        return ToolResult(ok=False, sentinel=ARXIV_UNAVAILABLE)
    refs = parse_openalex(payload, limit)
    if not refs:
        return ToolResult(ok=True, sentinel=NO_ARXIV_HITS)
    return ToolResult(ok=True, text=format_arxiv_hits(refs), refs=refs)


def _openalex_arxiv_id(work: dict[str, Any]) -> str:
    ids = work.get("ids") if isinstance(work.get("ids"), dict) else {}
    candidates = [str(ids.get("arxiv") or "")]
    loc = work.get("primary_location") if isinstance(work.get("primary_location"), dict) else {}
    candidates.append(str(loc.get("landing_page_url") or ""))
    for item in work.get("locations") or []:
        if isinstance(item, dict):
            candidates.append(str(item.get("landing_page_url") or ""))
    for raw in candidates:
        ident = normalize_arxiv_id(raw)
        if _looks_like_arxiv_id(ident):
            return ident
    return ""


def _looks_like_arxiv_id(ident: str) -> bool:
    text = (ident or "").strip()
    return bool(re.match(r"^(\d{4}\.\d{4,5}|[a-z-]+/\d{7})$", text, re.I))


def _openalex_abstract(inverted: Any) -> str:
    if not isinstance(inverted, dict):
        return ""
    ranked: list[tuple[int, str]] = []
    for word, indexes in inverted.items():
        if not isinstance(indexes, list):
            continue
        for index in indexes:
            if isinstance(index, int):
                ranked.append((index, str(word)))
    ranked.sort()
    return _collapse(" ".join(word for _, word in ranked))


def format_arxiv_hits(refs: list[ExternalRef]) -> str:
    blocks: list[str] = []
    for index, item in enumerate(refs, start=1):
        ident = item.identifier or ""
        year = item.year or ""
        header = f"[ext-{index}] {ident} year={year}".strip()
        lines = [header, item.title]
        authors = _authors_line(item.authors)
        if authors:
            lines.append(authors)
        snippet = (item.snippet or "").strip()
        if len(snippet) > 400:
            snippet = snippet[:400].rstrip() + "…"
        if snippet:
            lines.append(snippet)
        if item.url:
            lines.append(f"url={item.url}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def normalize_arxiv_id(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        return ""
    text = re.sub(r"^https?://(www\.)?arxiv\.org/(abs|pdf)/", "", text, flags=re.I)
    text = re.sub(r"^arxiv:", "", text, flags=re.I)
    text = text.replace(".pdf", "")
    text = text.split("/")[-1] if "arxiv.org" in text else text
    text = re.sub(r"v\d+$", "", text)
    return text.strip()


def _http_get(
    base_url: str,
    *,
    params: dict[str, str | int],
    timeout: httpx.Timeout,
    headers: dict[str, str],
    client: httpx.Client | None,
) -> httpx.Response:
    if client is not None:
        return client.get(
            base_url,
            params=params,
            timeout=timeout,
            headers=headers,
            follow_redirects=True,
        )
    last_exc: Exception | None = None
    try:
        with httpx.Client(
            trust_env=False,
            follow_redirects=True,
            timeout=timeout,
            headers=headers,
        ) as session:
            return session.get(base_url, params=params)
    except httpx.TimeoutException:
        raise
    except httpx.RequestError as exc:
        last_exc = exc
        logger.warning("arxiv_search direct connect failed err=%s; try system proxy", exc)
    try:
        with httpx.Client(
            trust_env=True,
            follow_redirects=True,
            timeout=timeout,
            headers=headers,
        ) as session:
            return session.get(base_url, params=params)
    except (httpx.TimeoutException, httpx.RequestError) as exc:
        if last_exc is not None:
            raise last_exc from exc
        raise


def _clamp_limit(raw: Any) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = DEFAULT_MAX_RESULTS
    return max(1, min(value, MAX_RESULTS_CAP))


def _search_pool(limit: int) -> int:
    return max(limit, min(SEARCH_POOL_CAP, limit * 2))


def _text(node: ET.Element | None) -> str:
    if node is None or node.text is None:
        return ""
    return str(node.text)


def _collapse(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _year(published: str) -> int | None:
    match = re.match(r"^(\d{4})", (published or "").strip())
    if not match:
        return None
    return int(match.group(1))


def _alternate_url(entry: ET.Element) -> str | None:
    for link in entry.findall(f"{ATOM}link"):
        rel = (link.get("rel") or "").strip()
        href = (link.get("href") or "").strip()
        if rel in {"", "alternate"} and href:
            return href.replace("http://", "https://", 1)
    return None


def _authors_line(authors: list[str]) -> str:
    if not authors:
        return ""
    if len(authors) > 4:
        return ", ".join(authors[:4]) + " et al."
    return ", ".join(authors)

"""arXiv 独立问答：只搜外部元数据，不进本篇 citations。"""

from __future__ import annotations

import re
from collections.abc import Callable

from dl_agent.config import Settings
from dl_agent.domain.models import AskTurn, ExternalRef, LibraryHit, Paper, PaperAnswer
from dl_agent.harness.complete import LlmRequestError
from dl_agent.knowledge.classify import looks_like_paper_title_heading
from dl_agent.knowledge.service import KnowledgeService
from dl_agent.mcp_gateway import build_gateway
from dl_agent.mcp_gateway.adapters.arxiv import (
    ARXIV_UNAVAILABLE,
    NO_ARXIV_HITS,
    build_arxiv_search_query,
    format_arxiv_hits,
    normalize_arxiv_id,
    strip_search_intent,
)
from dl_agent.mcp_gateway.base import McpGateway
from dl_agent.understand.citations import sanitize_answer_zh
from dl_agent.understand.library import _parse_json_object, _recent_history
from dl_agent.understand.progress import emit_phase

ChatFn = Callable[..., str]

NO_ARXIV_ANSWER = "arXiv 上没有找到相关论文。可以换个英文关键词或 arXiv id 再试。"
ARXIV_UNAVAILABLE_ANSWER = "暂时连不上 arXiv，请稍后重试。"
NEED_ARXIV_TOPIC = (
    "还没有可用的检索主题。请补充关键词、作者或 arXiv 分类（如 cs.LG、gr-qc），"
    "或打开一篇论文后再问「检索相关论文」。"
)
ONLY_SELF_ANSWER = "检索结果主要是当前这篇本身，没有可用的相关文献。可以再补一组任务或方法关键词。"
ARXIV_PROMPT_VERSION = "arxiv-ask-v2"
HIT_LIMIT = 5
MAX_REWRITE_QUERIES = 1

ARXIV_SYSTEM = """你是文献检索助手。根据 arXiv 返回的编号条目（标题、摘要、id）用中文回答。
这些是外部文献元数据，不是用户正在精读的那篇 PDF。

硬约束：
- 只用条目里出现的信息，不要编造实验数字。
- 不要用 [n] 当作当前 PDF 的页码引用。
- 已撤回、已移除或摘要写着 withdrawn/removed 的条目不要当有效文献。
- 若标注「本地已有」，可以提示用户打开本地副本精读。
- 中文讲解；术语、模型名保留英文。
- 只输出一个 JSON 对象：{"answer_zh":"..."}，不要 Markdown 围栏。"""

REWRITE_SYSTEM = """你是 arXiv 检索问句生成器。根据当前论文的标题与摘要，生成用于查找相关工作的英文检索句。

硬约束：
- 用该方向的通用任务、方法、模态术语，不要整句复述论文标题。
- 不要把论文自己的专有系统名或缩写当作唯一检索词。
- 每条 3–6 个英文词；用空格分词，不要写 Re-ID 这类连字符。
- 最多 2 条，互补不要同义重复。
- 只用标题和摘要里的主题，不要编造数据集或数字。
- 只输出一个 JSON 对象：{"queries":["..."]}，不要 Markdown 围栏。"""

_ARXIV_ID = re.compile(r"\b(\d{4}\.\d{4,5})(v\d+)?\b", re.IGNORECASE)


def ask_arxiv(
    knowledge: KnowledgeService,
    question: str,
    history: list[AskTurn] | None = None,
    *,
    settings: Settings,
    chat_fn: ChatFn,
    gateway: McpGateway | None = None,
    current_paper_id: str | None = None,
) -> PaperAnswer:
    question = (question or "").strip()
    if not question:
        raise ValueError("问题不能为空")
    emit_phase("arxiv", "正在检索 arXiv")
    version = getattr(settings, "arxiv_prompt_version", None) or ARXIV_PROMPT_VERSION
    model = settings.model_name
    gw = gateway or build_gateway(settings)
    query, arxiv_id = _search_args(question)
    title, abstract = _paper_brief(knowledge, current_paper_id)
    paper = knowledge.store.get_paper(current_paper_id) if current_paper_id else None
    rewritten: list[str] = []
    if not query and not arxiv_id:
        if not title and not abstract:
            return PaperAnswer(
                paper_id=current_paper_id or "arxiv",
                question=question,
                answer_zh=NEED_ARXIV_TOPIC,
                mode="arxiv",
                no_evidence=True,
                model=model,
                prompt_version=version,
            )
        try:
            rewritten = _rewrite_related_queries(
                chat_fn,
                settings,
                question,
                title=title,
                abstract=abstract,
            )
        except LlmRequestError:
            rewritten = []
        query_list = rewritten or _heuristic_related_queries(title, abstract)
    elif query:
        query_list = [query]
    else:
        query_list = []
    args: dict = {"max_results": HIT_LIMIT}
    if arxiv_id:
        args["arxiv_id"] = arxiv_id
        result = gw.call("arxiv_search", args)
        refs = list(result.refs or [])
        sentinel = result.sentinel
        ok = result.ok
    else:
        refs, sentinel, ok = _search_related(
            gw,
            query_list,
            title=title,
            paper=paper,
            limit=HIT_LIMIT,
        )
    answer_paper_id = current_paper_id or "arxiv"
    if sentinel == ARXIV_UNAVAILABLE or not ok:
        return PaperAnswer(
            paper_id=answer_paper_id,
            question=question,
            answer_zh=ARXIV_UNAVAILABLE_ANSWER,
            mode="arxiv",
            no_evidence=True,
            model=model,
            prompt_version=version,
        )
    if not refs or sentinel == NO_ARXIV_HITS:
        return PaperAnswer(
            paper_id=answer_paper_id,
            question=question,
            answer_zh=ONLY_SELF_ANSWER if not query and (title or abstract) else NO_ARXIV_ANSWER,
            mode="arxiv",
            no_evidence=True,
            model=model,
            prompt_version=version,
        )
    local_hits = match_local_refs(knowledge, refs, exclude_paper_id=current_paper_id)
    observation = format_arxiv_hits(refs)
    if rewritten or (not arxiv_id and query_list and not query):
        observation = (
            "[Search queries]\n"
            + "\n".join(f"- {item}" for item in query_list)
            + "\n\n"
            + observation
        )
    if local_hits:
        observation += "\n\n[Local copies]\n" + "\n".join(
            f"- paper_id={item.paper_id} title={item.title or item.filename}" for item in local_hits
        )
    raw = chat_fn(
        [
            {"role": "system", "content": ARXIV_SYSTEM},
            {
                "role": "user",
                "content": _user_prompt(question, observation, history, settings.ask_history_turns),
            },
        ],
        settings=settings,
    )
    payload = _parse_json_object(raw)
    answer = sanitize_answer_zh(
        str(payload.get("answer_zh") or "").strip() or "模型没有给出可用回答，请换个问法重试。"
    )
    return PaperAnswer(
        paper_id=answer_paper_id,
        question=question,
        answer_zh=answer,
        external_refs=refs,
        library_hits=local_hits,
        mode="arxiv",
        model=model,
        prompt_version=version,
    )


def match_local_refs(
    knowledge: KnowledgeService,
    refs: list[ExternalRef],
    *,
    exclude_paper_id: str | None = None,
) -> list[LibraryHit]:
    papers = [
        item
        for item in knowledge.list_papers()
        if item.status == "ready" and item.paper_id != exclude_paper_id
    ]
    out: list[LibraryHit] = []
    seen: set[str] = set()
    for ref in refs:
        for paper in papers:
            if paper.paper_id in seen:
                continue
            if not _ref_matches_paper(ref, paper):
                continue
            hit = knowledge.paper_card(paper.paper_id)
            if hit is None:
                continue
            hit.why = f"本地已有：与 arXiv「{ref.title}」标题或 id 相近"
            out.append(hit)
            seen.add(paper.paper_id)
    return out


_STOP = {
    "arxiv",
    "pdf",
    "http",
    "https",
    "www",
    "abs",
    "search",
    "papers",
    "paper",
    "related",
    "look",
    "find",
    "query",
    "literature",
    "please",
    "about",
    "some",
    "any",
    "the",
    "and",
    "for",
    "on",
    "in",
    "of",
    "to",
    "with",
}


def _search_args(question: str) -> tuple[str, str]:
    ident = ""
    match = _ARXIV_ID.search(question or "")
    if match:
        ident = normalize_arxiv_id(match.group(1))
    latin = [
        token
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_.+-]{1,}", question or "")
        if token.lower() not in _STOP
    ]
    query = " ".join(latin).strip()
    if not query:
        query = _chinese_topic(question)
    return query, ident


def _chinese_topic(question: str) -> str:
    leftover = strip_search_intent(question)
    leftover = re.sub(r"[A-Za-z][A-Za-z0-9_.+-]*", " ", leftover)
    leftover = re.sub(r"\s+", " ", leftover).strip()
    compact = re.sub(r"\s+", "", leftover)
    if len(compact) < 2:
        return ""
    return leftover


def _rewrite_related_queries(
    chat_fn: ChatFn,
    settings: Settings,
    question: str,
    *,
    title: str,
    abstract: str,
) -> list[str]:
    raw = chat_fn(
        [
            {"role": "system", "content": REWRITE_SYSTEM},
            {"role": "user", "content": _rewrite_user_prompt(question, title, abstract)},
        ],
        settings=settings,
    )
    payload = _parse_json_object(raw)
    out: list[str] = []
    for item in payload.get("queries") or []:
        query = re.sub(r"\s+", " ", str(item or "")).strip()
        if not query or not build_arxiv_search_query(query):
            continue
        if title and _titles_close(query, title):
            continue
        if query not in out:
            out.append(query)
        if len(out) >= MAX_REWRITE_QUERIES:
            break
    return out


def _heuristic_related_queries(title: str, abstract: str) -> list[str]:
    out: list[str] = []
    sentence = _first_sentence(_abstract_from_blob(abstract))
    if sentence and not (title and _titles_close(sentence, title)):
        out.append(sentence)
    rest = re.sub(r"^[A-Z][A-Z0-9+-]{2,16}\s*:\s*", "", title or "").strip()
    if rest and rest not in out and not (title and rest == title and not abstract):
        out.append(rest)
    elif title and title not in out and not out:
        out.append(title)
    return out[:MAX_REWRITE_QUERIES]


def _search_related(
    gateway: McpGateway,
    queries: list[str],
    *,
    title: str,
    paper: Paper | None,
    limit: int,
) -> tuple[list[ExternalRef], str | None, bool]:
    if not queries:
        return [], NO_ARXIV_HITS, True
    seen: set[str] = set()
    refs: list[ExternalRef] = []
    any_ok = False
    any_unavailable = False
    last_ok = True
    last_sentinel: str | None = NO_ARXIV_HITS
    for item in queries:
        result = gateway.call("arxiv_search", {"query": item, "max_results": limit})
        if result.sentinel == ARXIV_UNAVAILABLE or not result.ok:
            any_unavailable = True
            last_ok = result.ok
            last_sentinel = result.sentinel
            break
        any_ok = True
        for ref in result.refs or []:
            key = (ref.identifier or "") or _norm_title(ref.title)
            if not key or key in seen:
                continue
            if _is_current_ref(ref, title=title, paper=paper):
                continue
            seen.add(key)
            refs.append(ref)
            if len(refs) >= limit:
                return refs, None, True
    if refs:
        return refs, None, True
    if any_ok:
        return [], NO_ARXIV_HITS, True
    if any_unavailable:
        return [], last_sentinel, last_ok
    return [], NO_ARXIV_HITS, True


def _is_current_ref(ref: ExternalRef, *, title: str, paper: Paper | None) -> bool:
    if title and _titles_close(ref.title, title):
        return True
    if paper is None:
        return False
    return _ref_matches_paper(ref, paper)


def _paper_brief(knowledge: KnowledgeService, paper_id: str | None) -> tuple[str, str]:
    if not paper_id:
        return "", ""
    paper = knowledge.store.get_paper(paper_id)
    if paper is None:
        return "", ""
    sections = knowledge.store.get_sections(paper_id)
    title = (paper.title or "").strip()
    if not title and sections:
        heading = (sections[0].title or "").strip()
        if _usable_search_topic(heading):
            title = heading
    abstract = (paper.abstract or "").strip()
    if not abstract:
        match = next(
            (
                item
                for item in sections
                if item.kind == "abstract" or (item.title or "").strip().lower() == "abstract"
            ),
            None,
        )
        abstract = (match.text or "").strip() if match is not None else ""
    if not abstract and sections:
        abstract = _abstract_from_blob(sections[0].text or "")
    return title, abstract


def _abstract_from_blob(text: str) -> str:
    blob = re.sub(r"\s+", " ", text or "").strip()
    match = re.search(r"(?i)\babstract\b\s*[-—:]?\s*(.+)", blob)
    if match:
        return match.group(1).strip()
    return blob


def _rewrite_user_prompt(question: str, title: str, abstract: str) -> str:
    lines = ["[Current paper]"]
    if title:
        lines.append(f"title: {title}")
    snippet = re.sub(r"\s+", " ", abstract or "").strip()
    if len(snippet) > 1200:
        snippet = snippet[:1200].rstrip() + "…"
    if snippet:
        lines.append(f"abstract: {snippet}")
    lines.extend(["", "[User question]", question])
    return "\n".join(lines)


_NUMBERED_HEADING = re.compile(
    r"^(?:\d+(?:\.\d+)*|[IVXLC]{1,6}|[A-Z])[.\s:-]+",
    re.IGNORECASE,
)


def _usable_search_topic(text: str) -> bool:
    heading = re.sub(r"\s+", " ", text or "").strip()
    if len(heading) < 16:
        return False
    if looks_like_paper_title_heading(heading):
        return bool(build_arxiv_search_query(heading))
    if len(heading) >= 40 and not _NUMBERED_HEADING.match(heading):
        return bool(build_arxiv_search_query(heading))
    return False


def _first_sentence(text: str) -> str:
    blob = re.sub(r"\s+", " ", text or "").strip()
    if not blob:
        return ""
    sentence = re.split(r"(?<=[.!?])\s+", blob, maxsplit=1)[0]
    words = sentence.split()
    if len(words) > 18:
        sentence = " ".join(words[:18])
    return sentence


def _ref_matches_paper(ref: ExternalRef, paper: Paper) -> bool:
    ident = normalize_arxiv_id(ref.identifier or "")
    hay = " ".join(
        part
        for part in (paper.title, paper.filename, paper.abstract or "")
        if part
    ).lower()
    if ident and ident.lower() in hay.replace("arxiv:", ""):
        return True
    return _titles_close(ref.title, paper.title or "") or _titles_close(ref.title, paper.filename or "")


def _titles_close(left: str, right: str) -> bool:
    a = _norm_title(left)
    b = _norm_title(right)
    if not a or not b or len(a) < 8 or len(b) < 8:
        return False
    return a == b or a in b or b in a


def _norm_title(text: str) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", (text or "").lower())


def _user_prompt(
    question: str,
    observation: str,
    history: list[AskTurn] | None,
    history_turns: int,
) -> str:
    lines = ["[arXiv hits]", observation, ""]
    recent = _recent_history(history, history_turns)
    if recent:
        lines.append("[Recent history]")
        for turn in recent:
            prefix = "Q" if turn.role == "user" else "A"
            content = re.sub(r"\s+", " ", turn.content).strip()[:400]
            lines.append(f"{prefix}: {content}")
        lines.append("")
    lines.append("[User question]")
    lines.append(question)
    return "\n".join(lines)

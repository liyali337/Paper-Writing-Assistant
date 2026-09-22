"""精读 / 本地库 / arXiv：规则只给 kind 提示，不再互斥切整条流水线。"""

from __future__ import annotations

import re
from typing import Literal

AskRoute = Literal["close_read", "library", "arxiv"]

# 用户在问自己读过的库，而不是当前这篇的 related work。
_LIBRARY = re.compile(
    r"本地(知识)?库|知识库|"
    r"(我|之前|以前|曾经).{0,12}(研读|读过|看过|上传)|"
    r"(研读|读过|看过)的(论文|文章)|"
    r"库里|"
    r"(其它|其他|别的)(那些)?论文|"
    r"还有(没有)?类似|"
    r"有没有(研读|读过)|"
    r"读过哪些|"
    r"\b(my library|papers i(?:'ve| have) (?:read|studied)|have i (?:read|studied))\b",
    re.IGNORECASE,
)

# 去网上 / arXiv 搜，而不是读当前这篇的 related work 节。
_ARXIV = re.compile(
    r"arxiv|"
    r"检索.{0,10}(相关)?(论文|文献)|"
    r"(搜|查)(一下)?(arxiv|相关论文)|"
    r"(网上|文献库).{0,8}(论文|文献)|"
    r"(某一|这个|该|某)方向的论文|"
    r"找.{0,6}篇.{0,8}(论文|工作)|"
    r"后续工作.{0,8}(论文|arxiv|网上)|"
    r"\b(search (?:arxiv|papers)|related papers|on arxiv|look up papers)\b",
    re.IGNORECASE,
)

# 库信号已经命中时，只有这些才额外加 arXiv，避免「检索本地库相关论文」双开。
_ARXIV_EXPLICIT = re.compile(
    r"arxiv|"
    r"(网上|文献库).{0,8}(论文|文献)|"
    r"\b(search arxiv|on arxiv|look up papers|related papers)\b",
    re.IGNORECASE,
)

# 身份 / 用法，不需要进检索图。
_OBVIOUS_CHAT = re.compile(
    r"你是谁|你是干什么|你是做什么|你能做什么|"
    r"(怎么用|如何使用)(这个)?(面板|助手)|"
    r"(你的|这个助手的)(功能|作用)是什么|"
    r"\bwho are you\b|\bwhat (?:can|do) you do\b",
    re.IGNORECASE,
)


# 明显在问当前这篇，即使夹了「相关」。
_STAY_CLOSE = re.compile(
    r"本文相关工作|这篇的相关工作|原文(里|中)?的相关|"
    r"这一节|这一章|这一段|"
    r"这篇(论文)?(的|在|如何|怎么|核心|方法|实验|公式|损失|结论)",
    re.IGNORECASE,
)


def is_obvious_chat(question: str) -> bool:
    text = (question or "").strip()
    if not text:
        return False
    return bool(_OBVIOUS_CHAT.search(text))


def suggest_task_kinds(question: str) -> list[AskRoute]:
    """可多选。无库/arXiv 信号时默认精读；双边问句会同时带上。"""
    text = (question or "").strip()
    if not text:
        return ["close_read"]
    stay = bool(_STAY_CLOSE.search(text))
    library = bool(_LIBRARY.search(text))
    if library:
        arxiv = bool(_ARXIV_EXPLICIT.search(text))
    else:
        arxiv = bool(_ARXIV.search(text))
    kinds: list[AskRoute] = []
    if stay or (not library and not arxiv):
        kinds.append("close_read")
    if library:
        kinds.append("library")
    if arxiv:
        kinds.append("arxiv")
    return kinds or ["close_read"]


def primary_ask_mode(question: str | None = None, *, kinds: list[str] | None = None) -> AskRoute:
    selected = kinds if kinds is not None else suggest_task_kinds(question or "")
    if "close_read" in selected:
        return "close_read"
    if selected:
        return selected[0]
    return "close_read"


def classify_ask_mode(question: str) -> AskRoute:
    """主模式：有精读任务则 close_read，否则取唯一的库/arXiv。"""
    return primary_ask_mode(question)

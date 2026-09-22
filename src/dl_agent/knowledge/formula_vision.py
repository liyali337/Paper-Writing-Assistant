"""用多模态模型看公式裁图，把坏 LaTeX 重识别成可给 KaTeX 用的源码。"""

from __future__ import annotations

import base64
import json
import logging
import re
from collections.abc import Callable
from typing import Any

from dl_agent.config import Settings
from dl_agent.knowledge.formula_latex import (
    is_display_formula,
    latex_needs_vision,
    latex_score,
    normalize_latex,
)
from dl_agent.knowledge.layout import LayoutItem

logger = logging.getLogger(__name__)

ChatFn = Callable[..., str]

_VISION_HINT = re.compile(
    r"(?:-vl\b|\bvl-|\bvision\b|gpt-4o|gpt-4\.1|gpt-5|gemini|"
    r"qwen-vl|qwen2\.5-vl|qwen3-vl|deepseek-flash|deepseek-v4-flash)",
    re.IGNORECASE,
)

SYSTEM_PROMPT = """You transcribe formulas from academic PDF crops into KaTeX-compatible LaTeX.

Return ONLY JSON: {"latex":"..."}
Rules:
- Copy the formula in the image exactly. Do not invent symbols, terms, or limits.
- Do not wrap the latex in $ or $$.
- Use \\tag{n} for a trailing equation number like (3), not \\label.
- Convert unicode operators to LaTeX (\\oplus, \\in, \\times, \\cdot, \\leq, \\sum).
- Use \\mathrm{...} for multi-letter subscripts such as inv, cls, CT.
- Scripts must be valid TeX: no double superscripts or double subscripts.
- If the image is not a formula, return {"latex":""}."""


def resolve_formula_vision_model(settings: Settings) -> str | None:
    explicit = settings.formula_vision_model.strip()
    if explicit:
        return explicit
    name = settings.model_name.strip()
    if name and _VISION_HINT.search(name):
        return name
    base = (settings.formula_vision_base_url or settings.openai_base_url).lower()
    if "dashscope" in base or "aliyuncs.com" in base:
        return "qwen-vl-plus"
    if "api.deepseek.com" in base:
        return "deepseek-flash"
    return None


def repair_formula_items(
    items: list[LayoutItem],
    *,
    settings: Settings,
    chat_fn: ChatFn | None = None,
) -> list[LayoutItem]:
    """就地改 formula 项的 text；失败则保留原 LaTeX。图只作为模型输入。"""
    if not settings.formula_vision_enabled:
        return items
    targets = [
        index
        for index, item in enumerate(items)
        if _needs_formula_vision(items, index)
    ]
    if not targets:
        return items
    model = resolve_formula_vision_model(settings)
    key = (settings.formula_vision_api_key or settings.openai_api_key).strip()
    if chat_fn is None:
        if not model or not key:
            logger.info("formula vision skipped: no vision model or api key")
            return items
    elif not model:
        model = settings.model_name.strip() or "vision"
    limit = max(0, settings.formula_vision_max)
    attempted = 0
    succeeded = 0
    for index in targets:
        if attempted >= limit:
            logger.info("formula vision hit max=%s", limit)
            break
        item = items[index]
        attempted += 1
        try:
            latex = _transcribe_formula(
                item,
                context=_neighbor_text(items, index),
                settings=settings,
                model=model,
                key=key,
                chat_fn=chat_fn,
            )
        except Exception:
            logger.warning("formula vision item failed page=%s", item.page, exc_info=True)
            continue
        if not latex or latex_needs_vision(latex):
            continue
        old = item.text
        if (
            old
            and not latex_needs_vision(old)
            and latex_score(latex) < latex_score(old)
        ):
            continue
        item.text = latex
        _strip_following_leak(items, index)
        succeeded += 1
    logger.info(
        "formula vision done attempted=%s succeeded=%s remaining=%s",
        attempted,
        succeeded,
        max(0, len(targets) - attempted),
    )
    return items


def _needs_formula_vision(items: list[LayoutItem], index: int) -> bool:
    item = items[index]
    if item.kind != "formula" or not item.image_bytes:
        return False
    if latex_needs_vision(item.text):
        return True
    # 独立展示式：启发式过不了「能渲染但抽错」。行内 $x$ 不送。
    if item.bbox and is_display_formula(item):
        return True
    nxt = items[index + 1] if index + 1 < len(items) else None
    return bool(nxt is not None and nxt.kind == "text" and _has_leaked_formula(nxt.text))


def _has_leaked_formula(text: str) -> bool:
    """后文里混进了展示公式的编号/赋值，不是 where 里的行内集合记号。"""
    blob = text or ""
    return bool(
        re.search(r"(?<![$])\\tag\{", blob)
        or re.search(r"[A-Za-z]_\{[A-Za-z]{2,}\}\s*=", blob)
    )


def _strip_following_leak(items: list[LayoutItem], index: int) -> None:
    if index + 1 >= len(items):
        return
    nxt = items[index + 1]
    if nxt.kind != "text" or not _has_leaked_formula(nxt.text):
        return
    nxt.text = _strip_leaked_formula(nxt.text)


_LEAKED_INLINE = re.compile(
    r"\$\\(?:times|cdot|leq|geq|in|oplus|otimes|tag\{[^}]+\})\$"
)
_LEAKED_TAG = re.compile(r"(?<![$])\\tag\{[^}]+\}")
_LEAKED_ASSIGN = re.compile(
    r"(?:^|\n)\s*[A-Za-z][A-Za-z0-9]*(?:_\{[^{}]+\})?(?:\^\{[^{}]+\})?\s*=.{0,180}?(?:\\in|∈).{0,80}",
    re.S,
)


def _strip_leaked_formula(text: str) -> str:
    blob = text or ""
    blob = _LEAKED_ASSIGN.sub(" ", blob, count=1)
    blob = _LEAKED_INLINE.sub(" ", blob)
    blob = _LEAKED_TAG.sub(" ", blob)
    blob = re.sub(
        r"\$\\?(?:times|leq|geq|in|oplus|[A-Za-z])\$",
        " ",
        blob,
    )
    blob = re.sub(r"[ \t]{2,}", " ", blob)
    blob = re.sub(r"\n{3,}", "\n\n", blob)
    return blob.strip()


def _transcribe_formula(
    item: LayoutItem,
    *,
    context: str,
    settings: Settings,
    model: str,
    key: str,
    chat_fn: ChatFn | None,
) -> str:
    png = item.image_bytes or b""
    if not png:
        return ""
    broken = normalize_latex(item.text)
    user_text = (
        "Broken OCR latex (may be empty or wrong):\n"
        f"{broken or '(empty)'}\n\n"
        "Nearby text:\n"
        f"{context or '(none)'}\n\n"
        "Transcribe the formula shown in the image."
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": _png_data_url(png)},
                },
                {"type": "text", "text": user_text},
            ],
        },
    ]
    if chat_fn is not None:
        try:
            raw = chat_fn(
                messages,
                settings=settings,
                model=model,
                base_url=settings.formula_vision_base_url or None,
                api_key=key or None,
                timeout_s=settings.formula_vision_timeout_s,
            )
        except TypeError:
            raw = chat_fn(messages)
    else:
        from dl_agent.harness.complete import chat

        raw = chat(
            messages,
            settings=settings,
            model=model,
            base_url=settings.formula_vision_base_url or None,
            api_key=key or None,
            timeout_s=settings.formula_vision_timeout_s,
        )
    return _parse_latex_response(raw)


def _png_data_url(png: bytes) -> str:
    blob = png
    if len(blob) > 1_500_000:
        blob = _shrink_png(blob)
    encoded = base64.b64encode(blob).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _shrink_png(png: bytes) -> bytes:
    import io

    from PIL import Image

    image = Image.open(io.BytesIO(png))
    image.thumbnail((1280, 1280))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _neighbor_text(items: list[LayoutItem], index: int) -> str:
    pieces: list[str] = []
    for offset in (-2, -1, 1, 2):
        pos = index + offset
        if pos < 0 or pos >= len(items):
            continue
        other = items[pos]
        if other.kind not in {"text", "heading", "caption"}:
            continue
        blob = " ".join(other.text.split())
        if blob:
            pieces.append(blob[:240])
    return " | ".join(pieces)[:500]


def _parse_latex_response(raw: str) -> str:
    data = _parse_json_object(raw)
    latex = normalize_latex(str(data.get("latex") or ""))
    if not latex:
        return ""
    if re.search(r"\b[A-Za-z]{12,}\b", re.sub(r"\\[A-Za-z]+", " ", latex)):
        return ""
    return latex


def _parse_json_object(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    fence = re.match(r"^```(?:json)?\s*([\s\S]*?)```\s*$", text)
    if fence:
        text = fence.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            return {}
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    return data if isinstance(data, dict) else {}

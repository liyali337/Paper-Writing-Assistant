from dl_agent.knowledge.adapters.pdf.types import ParseResult

__all__ = ["ParseResult", "parse_pdf"]


def parse_pdf(
    pdf_path: str,
    *,
    timeout_s: float = 120,
    images_scale: float = 2.0,
    min_chars: int = 800,
    formula_enrichment: bool = False,
) -> ParseResult:
    """主路径 Docling，失败或文本过少则降级 PyMuPDF。"""
    import logging

    from dl_agent.knowledge.pdf_io import ParseError

    logger = logging.getLogger(__name__)
    docling_error: Exception | None = None
    sparse: ParseResult | None = None

    try:
        from dl_agent.knowledge.adapters.pdf import docling_adapter

        try:
            result = docling_adapter.parse_pdf(
                pdf_path,
                timeout_s=timeout_s,
                images_scale=images_scale,
                formula_enrichment=formula_enrichment,
            )
        except Exception as formula_exc:
            if not formula_enrichment:
                raise
            logger.warning(
                "docling formula enrichment failed, retry without VLM: %s",
                formula_exc,
            )
            result = docling_adapter.parse_pdf(
                pdf_path,
                timeout_s=timeout_s,
                images_scale=images_scale,
                formula_enrichment=False,
            )
        if result.char_count >= min_chars:
            try:
                from dl_agent.knowledge.adapters.pdf.pymupdf_adapter import clip_missing_captions

                result.items = clip_missing_captions(pdf_path, result.items)
            except Exception:
                logger.warning("caption clip fallback failed", exc_info=True)
            try:
                from dl_agent.knowledge.adapters.pdf.pymupdf_adapter import clip_formula_items

                result.items = clip_formula_items(pdf_path, result.items)
            except Exception:
                logger.warning("formula clip fallback failed", exc_info=True)
            try:
                from dl_agent.knowledge.formula_latex import inject_inline_math

                result.items = inject_inline_math(pdf_path, result.items)
            except Exception:
                logger.warning("inline math inject failed", exc_info=True)
            return result
        sparse = result
        logger.warning("docling text sparse chars=%s, fallback pymupdf", result.char_count)
    except ImportError as exc:
        docling_error = exc
        logger.info("docling not installed, using pymupdf")
    except Exception as exc:
        docling_error = exc
        logger.warning("docling parse failed: %s", exc)

    try:
        from dl_agent.knowledge.adapters.pdf import pymupdf_adapter

        return pymupdf_adapter.parse_pdf(pdf_path)
    except Exception as exc:
        if sparse is not None:
            return sparse
        raise ParseError("pdf parse failed") from (exc if docling_error is None else docling_error)

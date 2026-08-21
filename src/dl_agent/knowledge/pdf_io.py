from __future__ import annotations

import hashlib
import logging

logger = logging.getLogger(__name__)

PDF_MAGIC = b"%PDF"


class InvalidPdfError(ValueError):
    def __init__(self, message: str):
        super().__init__(message)
        self.code = "invalid_pdf"


class ParseError(RuntimeError):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_pdf_bytes(data: bytes, *, max_bytes: int) -> None:
    if len(data) > max_bytes:
        raise InvalidPdfError(f"PDF 超过大小上限（{max_bytes} 字节）")
    if len(data) < 8 or not data.lstrip().startswith(PDF_MAGIC):
        raise InvalidPdfError("不是有效的 PDF 文件")
    try:
        import pymupdf
    except ImportError as exc:
        raise InvalidPdfError("缺少 PyMuPDF，无法校验 PDF") from exc
    try:
        document = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise InvalidPdfError("无法打开 PDF") from exc
    try:
        if document.is_encrypted or getattr(document, "needs_pass", False):
            unlocked = document.authenticate("")
            if not unlocked:
                raise InvalidPdfError("PDF 已加密，无法打开")
        if document.page_count < 1:
            raise InvalidPdfError("PDF 不含页面")
    finally:
        document.close()

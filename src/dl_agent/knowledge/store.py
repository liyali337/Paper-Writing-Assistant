from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from pydantic import ValidationError

from dl_agent.domain.models import Figure, Paper, PaperTranslation, Section


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


class FilePaperStore:
    """本地目录对象存储 + JSON 元数据。键按 S3 风格 papers/{id}/..."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _paper_dir(self, paper_id: str, *, create: bool = False) -> Path:
        path = self.root / "papers" / paper_id
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    def _index_path(self) -> Path:
        return self.root / "index.json"

    def _load_index(self) -> dict[str, str]:
        path = self._index_path()
        if not path.exists() or path.stat().st_size == 0:
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def _save_index(self, index: dict[str, str]) -> None:
        _atomic_write_text(
            self._index_path(),
            json.dumps(index, ensure_ascii=False, indent=2),
        )

    def get_paper_id_by_sha(self, sha256: str) -> str | None:
        with self._lock:
            return self._load_index().get(sha256)

    def index_sha(self, sha256: str, paper_id: str) -> None:
        with self._lock:
            index = self._load_index()
            index[sha256] = paper_id
            self._save_index(index)

    def save_source_pdf(self, paper_id: str, data: bytes) -> Path:
        path = self._paper_dir(paper_id, create=True) / "source.pdf"
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)
        return path

    def source_pdf_path(self, paper_id: str) -> Path:
        return self._paper_dir(paper_id) / "source.pdf"

    def save_paper(self, paper: Paper) -> None:
        path = self._paper_dir(paper.paper_id, create=True) / "paper.json"
        _atomic_write_text(path, paper.model_dump_json(indent=2))

    def get_paper(self, paper_id: str) -> Paper | None:
        path = self._paper_dir(paper_id) / "paper.json"
        if not path.exists() or path.stat().st_size == 0:
            return None
        try:
            return Paper.model_validate_json(path.read_text(encoding="utf-8"))
        except (ValidationError, json.JSONDecodeError):
            return None

    def save_sections(self, paper_id: str, sections: list[Section]) -> None:
        path = self._paper_dir(paper_id, create=True) / "sections.json"
        payload = [item.model_dump() for item in sections]
        _atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2))

    def get_sections(self, paper_id: str) -> list[Section]:
        path = self._paper_dir(paper_id) / "sections.json"
        if not path.exists() or path.stat().st_size == 0:
            return []
        raw = json.loads(path.read_text(encoding="utf-8"))
        return [Section.model_validate(item) for item in raw]

    def delete_figures(self, paper_id: str) -> None:
        fig_dir = self._paper_dir(paper_id, create=True) / "figures"
        if fig_dir.exists():
            for child in fig_dir.iterdir():
                child.unlink(missing_ok=True)
        meta = self._paper_dir(paper_id) / "figures.json"
        if meta.exists():
            meta.unlink()

    def save_figures(
        self, paper_id: str, figures: list[Figure], pngs: dict[str, bytes]
    ) -> None:
        fig_dir = self._paper_dir(paper_id, create=True) / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        for figure_id, blob in pngs.items():
            (fig_dir / f"{figure_id}.png").write_bytes(blob)
        path = self._paper_dir(paper_id) / "figures.json"
        payload = [item.model_dump() for item in figures]
        _atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2))

    def get_figures(self, paper_id: str, section_id: str | None = None) -> list[Figure]:
        path = self._paper_dir(paper_id) / "figures.json"
        if not path.exists() or path.stat().st_size == 0:
            return []
        raw = json.loads(path.read_text(encoding="utf-8"))
        figures = [Figure.model_validate(item) for item in raw]
        if section_id:
            figures = [item for item in figures if item.section_id == section_id]
        return figures

    def figure_png_path(self, paper_id: str, figure_id: str) -> Path | None:
        path = self._paper_dir(paper_id) / "figures" / f"{figure_id}.png"
        if path.exists():
            return path
        return None

    def append_audit(self, record: dict) -> None:
        path = self.root / "audit.jsonl"
        with self._lock:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def save_translation(self, translation: PaperTranslation) -> None:
        path = self._paper_dir(translation.paper_id, create=True) / "translations.json"
        _atomic_write_text(path, translation.model_dump_json(indent=2))

    def get_translation(self, paper_id: str) -> PaperTranslation | None:
        path = self._paper_dir(paper_id) / "translations.json"
        if not path.exists() or path.stat().st_size == 0:
            return None
        try:
            return PaperTranslation.model_validate_json(path.read_text(encoding="utf-8"))
        except (ValidationError, json.JSONDecodeError):
            return None

    def delete_translation(self, paper_id: str) -> None:
        path = self._paper_dir(paper_id) / "translations.json"
        if path.exists():
            path.unlink()

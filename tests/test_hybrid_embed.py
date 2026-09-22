import os
import sys
import types

from dl_agent.config import Settings
from dl_agent.knowledge.adapters.vector import hybrid_embed


class _Row:
    def __init__(self, values: list[float]):
        self._values = values

    def tolist(self) -> list[float]:
        return list(self._values)


class _Batch:
    def __init__(self, rows: list[list[float]]):
        self._rows = [_Row(row) for row in rows]

    def tolist(self) -> list[list[float]]:
        return [row.tolist() for row in self._rows]

    def __getitem__(self, index: int) -> _Row:
        return self._rows[index]


def test_sentence_transformer_uses_local_cache_first(monkeypatch) -> None:
    calls: list[bool] = []

    class FakeST:
        def __init__(self, name: str, local_files_only: bool = False, **kwargs):
            calls.append(local_files_only)
            if not local_files_only:
                raise AssertionError("should not download when cache hits")

        def encode(self, texts, normalize_embeddings=True):
            return _Batch([[0.1, 0.2] for _ in texts])

    fake_mod = types.ModuleType("sentence_transformers")
    fake_mod.SentenceTransformer = FakeST
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_mod)

    found = hybrid_embed._try_sentence_transformer("sentence-transformers/all-mpnet-base-v2")
    assert found is not None
    documents, query = found
    assert documents(["hello"]) == [[0.1, 0.2]]
    assert query("hello") == [0.1, 0.2]
    assert calls == [True]
    assert os.environ.get("HF_HUB_OFFLINE") != "1"


def test_apply_hf_endpoint_sets_env(monkeypatch) -> None:
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    hybrid_embed._apply_hf_endpoint(Settings(hf_endpoint="https://hf-mirror.com"))
    assert os.environ["HF_ENDPOINT"] == "https://hf-mirror.com"

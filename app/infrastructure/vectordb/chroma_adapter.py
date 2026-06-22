\
\
\
\

from __future__ import annotations

from typing import Any

from loguru import logger

try:
    import chromadb
except ImportError:                    
    chromadb = None                            

class _Entity:

    def __init__(self, payload: dict[str, Any]) -> None:
        self._d = payload

    def to_dict(self) -> dict[str, Any]:
        return dict(self._d)

    def get(self, key: str, default: Any = None) -> Any:
        return self._d.get(key, default)

class _Hit:

    def __init__(self, hit_id: str, distance: float, entity: dict[str, Any]) -> None:
        self.id = hit_id
        self.distance = distance
        self.entity = _Entity(entity)

class ChromaCollectionAdapter:

    def __init__(self, chroma_collection: Any) -> None:
        self._coll = chroma_collection

    @property
    def name(self) -> str:
        return self._coll.name

    def count(self) -> int:
        return int(self._coll.count())

    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        if not ids:
            return
        meta = metadatas or [{"_": "_"} for _ in ids]
        self._coll.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=meta)

    def search(
        self,
        data: list[list[float]],
        anns_field: str = "embedding",
        param: dict[str, Any] | None = None,
        limit: int = 10,
        output_fields: list[str] | None = None,
        **kwargs: Any,
    ) -> list[list[_Hit]]:

        results: list[list[_Hit]] = []
        for vec in data:
            r = self._coll.query(query_embeddings=[vec], n_results=limit)
            ids = (r.get("ids") or [[]])[0]
            docs = (r.get("documents") or [[]])[0]
            dists = (r.get("distances") or [[]])[0]
            metas = (r.get("metadatas") or [[]])[0]
            hits: list[_Hit] = []
            for i, _id in enumerate(ids):
                doc = docs[i] if i < len(docs) else ""
                dist = dists[i] if i < len(dists) else 0.0
                entity: dict[str, Any] = {"id": _id, "text": doc}
                if i < len(metas) and metas[i]:
                    entity.update({k: v for k, v in metas[i].items() if k != "_"})
                hits.append(_Hit(hit_id=str(_id), distance=float(dist), entity=entity))
            results.append(hits)
        return results

def get_or_create_collection(persist_dir: str, name: str) -> ChromaCollectionAdapter:

    if chromadb is None:
        raise RuntimeError("chromadb 未安装，请 pip install chromadb")
    client = chromadb.PersistentClient(path=persist_dir)
    coll = client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info("Chroma 集合 [{}] 就绪 path={}", name, persist_dir)
    return ChromaCollectionAdapter(coll)

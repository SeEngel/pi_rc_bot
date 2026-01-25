from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

import numpy as np


class MemoryError(RuntimeError):
    pass


def _utc_now_epoch() -> float:
    return time.time()


def _dt_to_iso_z(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)
    # Keep seconds precision (stable, readable)
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_timestamp(ts: Any | None) -> tuple[float, str]:
    """Parse timestamp input into (epoch_seconds, iso_z).

    Accepted formats:
    - None: now
    - number (int/float): epoch seconds
    - string:
      - epoch seconds "1730000000" / "1730000000.5"
      - ISO 8601, e.g. "2026-01-12T10:20:30Z" or with offset
    - datetime
    """
    if ts is None or ts == "":
        now = datetime.now(timezone.utc)
        return now.timestamp(), _dt_to_iso_z(now)

    if isinstance(ts, datetime):
        dt = ts
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        dt = dt.astimezone(timezone.utc)
        return dt.timestamp(), _dt_to_iso_z(dt)

    if isinstance(ts, (int, float)):
        dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
        return float(ts), _dt_to_iso_z(dt)

    if isinstance(ts, str):
        s = ts.strip()
        if not s:
            now = datetime.now(timezone.utc)
            return now.timestamp(), _dt_to_iso_z(now)
        # epoch string
        try:
            val = float(s)
            dt = datetime.fromtimestamp(val, tz=timezone.utc)
            return val, _dt_to_iso_z(dt)
        except Exception:
            pass
        # ISO
        try:
            # Accept trailing Z
            if s.endswith("Z"):
                s2 = s[:-1] + "+00:00"
            else:
                s2 = s
            dt = datetime.fromisoformat(s2)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            dt = dt.astimezone(timezone.utc)
            return dt.timestamp(), _dt_to_iso_z(dt)
        except Exception as exc:
            raise MemoryError(f"Invalid timestamp format: {ts!r}") from exc

    raise MemoryError(f"Invalid timestamp type: {type(ts).__name__}")


def _atomic_write_text(path: str, text: str) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def _atomic_write_bytes(path: str, data: bytes) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _as_str_list(val: Any | None) -> list[str]:
    if val is None:
        return []
    if isinstance(val, list):
        out: list[str] = []
        for x in val:
            if x is None:
                continue
            s = str(x).strip()
            if s:
                out.append(s)
        return out
    if isinstance(val, str):
        s = val.strip()
        return [s] if s else []
    return [str(val)]


def _normalize(vec: np.ndarray) -> np.ndarray:
    vec = vec.astype(np.float32, copy=False)
    n = float(np.linalg.norm(vec))
    if n <= 0:
        return vec
    return vec / n


@dataclass(frozen=True)
class OpenAIEmbeddingSettings:
    model: str
    base_url: str | None = None
    timeout_seconds: float = 30.0


class OpenAIEmbedder:
    def __init__(self, settings: OpenAIEmbeddingSettings):
        self.settings = settings

    def embed(self, text: str) -> np.ndarray:
        text = (text or "").strip()
        if not text:
            raise MemoryError("Cannot embed empty text")

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise MemoryError("OPENAI_API_KEY is not set")

        try:
            from openai import OpenAI
        except Exception as exc:
            raise MemoryError("openai package not installed") from exc

        kwargs: dict[str, Any] = {"api_key": api_key}
        base_url = (self.settings.base_url or "").strip()
        env_base_url = (os.environ.get("OPENAI_BASE_URL") or "").strip()
        if env_base_url:
            base_url = env_base_url
        if base_url:
            kwargs["base_url"] = base_url

        client = OpenAI(**kwargs)
        try:
            res = client.embeddings.create(
                model=self.settings.model,
                input=text,
                timeout=float(self.settings.timeout_seconds or 30.0),
            )
        except TypeError:
            # Some versions of the SDK may not accept timeout here; fall back.
            res = client.embeddings.create(model=self.settings.model, input=text)

        if not res.data:
            raise MemoryError("No embedding returned")
        emb = res.data[0].embedding
        return np.asarray(emb, dtype=np.float32)

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Embed multiple texts in a single API call (more efficient)."""
        texts = [t.strip() for t in texts if t and t.strip()]
        if not texts:
            return []

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise MemoryError("OPENAI_API_KEY is not set")

        try:
            from openai import OpenAI
        except Exception as exc:
            raise MemoryError("openai package not installed") from exc

        kwargs: dict[str, Any] = {"api_key": api_key}
        base_url = (self.settings.base_url or "").strip()
        env_base_url = (os.environ.get("OPENAI_BASE_URL") or "").strip()
        if env_base_url:
            base_url = env_base_url
        if base_url:
            kwargs["base_url"] = base_url

        client = OpenAI(**kwargs)
        try:
            res = client.embeddings.create(
                model=self.settings.model,
                input=texts,
                timeout=float(self.settings.timeout_seconds or 30.0),
            )
        except TypeError:
            res = client.embeddings.create(model=self.settings.model, input=texts)

        if not res.data:
            raise MemoryError("No embeddings returned")

        # Sort by index to ensure correct order
        sorted_data = sorted(res.data, key=lambda x: x.index)
        return [np.asarray(d.embedding, dtype=np.float32) for d in sorted_data]

    def model_id(self) -> str:
        """Return a unique identifier for the current embedding model."""
        return self.settings.model


def _compute_model_hash(model_id: str) -> str:
    """Compute a short hash for model identification."""
    return hashlib.sha256(model_id.encode()).hexdigest()[:16]


class MemoryStore:
    """Persistent two-tier embedding store (short-term + long-term) with tag-based filtering."""

    def __init__(
        self,
        *,
        data_dir: str,
        embedder: OpenAIEmbedder,
        short_time_seconds: float = 3600.0,
        long_time_seconds: float = 2592000.0,
        prune_older_than_long_time: bool = False,
        # Legacy: global cap across BOTH tiers. If per-tier caps are provided,
        # this is ignored.
        max_memory_strings: int | None = 1000,
        # Preferred: independent caps per tier.
        max_short_memory_strings: int | None = None,
        max_long_memory_strings: int | None = None,
    ):
        self.data_dir = os.path.abspath(data_dir)
        self.embedder = embedder
        self.short_time_seconds = float(short_time_seconds)
        self.long_time_seconds = float(long_time_seconds)
        self.prune_older_than_long_time = bool(prune_older_than_long_time)

        self.max_memory_strings = int(max_memory_strings) if max_memory_strings is not None else None
        self.max_short_memory_strings = (
            int(max_short_memory_strings) if max_short_memory_strings is not None else None
        )
        self.max_long_memory_strings = int(max_long_memory_strings) if max_long_memory_strings is not None else None

        self._short_meta: list[dict[str, Any]] = []
        self._long_meta: list[dict[str, Any]] = []

        # Content vectors: shape (dim, n) where each column is a unit vector.
        self._short_mat: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self._long_mat: np.ndarray = np.zeros((0, 0), dtype=np.float32)

        # Tag system: global tag registry with vectors
        self._tag_to_idx: dict[str, int] = {}  # tag_name -> index in tag_vectors
        self._tag_vectors: np.ndarray = np.zeros((0, 0), dtype=np.float32)  # shape (dim, n_tags)
        self._stored_model_id: str | None = None  # Model ID when vectors were created

        self._load()
        # Check if model changed - if so, rebuild all vectors
        self._check_and_rebuild_vectors_if_needed()
        # Repartition on startup (short memories may have aged into long).
        self.repartition(now_epoch=_utc_now_epoch())
        self._enforce_limits(now_epoch=_utc_now_epoch())
        self._save()

    def _check_and_rebuild_vectors_if_needed(self) -> None:
        """Check if embedding model changed and rebuild vectors if necessary."""
        current_model = self.embedder.model_id()
        
        if self._stored_model_id is None:
            # First run or legacy data - need to build tag vectors
            print(f"[MemoryStore] No stored model_id found, will build tag vectors")
            self._stored_model_id = current_model
            self._rebuild_tag_vectors()
            return

        if self._stored_model_id != current_model:
            print(f"[MemoryStore] Model changed from '{self._stored_model_id}' to '{current_model}'")
            print(f"[MemoryStore] Rebuilding ALL vectors...")
            self._stored_model_id = current_model
            self._rebuild_all_vectors()
            return

        # Model is same, but check if any tags are missing vectors
        all_tags = self._collect_all_tags()
        missing_tags = [t for t in all_tags if t not in self._tag_to_idx]
        if missing_tags:
            print(f"[MemoryStore] Found {len(missing_tags)} tags without vectors, adding them...")
            self._add_tag_vectors(missing_tags)

    def _collect_all_tags(self) -> set[str]:
        """Collect all unique tags from all memories."""
        tags: set[str] = set()
        for m in self._short_meta:
            for t in m.get("tags", []):
                if t and isinstance(t, str):
                    tags.add(t.strip().lower())
        for m in self._long_meta:
            for t in m.get("tags", []):
                if t and isinstance(t, str):
                    tags.add(t.strip().lower())
        return tags

    def _rebuild_tag_vectors(self) -> None:
        """Build tag vectors for all existing tags."""
        all_tags = sorted(self._collect_all_tags())
        if not all_tags:
            print("[MemoryStore] No tags to vectorize")
            return
        
        print(f"[MemoryStore] Vectorizing {len(all_tags)} tags...")
        self._tag_to_idx = {}
        self._tag_vectors = np.zeros((0, 0), dtype=np.float32)
        self._add_tag_vectors(all_tags)

    def _rebuild_all_vectors(self) -> None:
        """Rebuild ALL vectors (content + tags) - called when model changes."""
        # Rebuild content vectors for short-term
        if self._short_meta:
            print(f"[MemoryStore] Rebuilding {len(self._short_meta)} short-term content vectors...")
            contents = [m.get("content", "") for m in self._short_meta]
            vectors = self.embedder.embed_batch(contents)
            if vectors:
                self._short_mat = np.stack([_normalize(v) for v in vectors], axis=1)
            else:
                self._short_mat = np.zeros((0, 0), dtype=np.float32)

        # Rebuild content vectors for long-term
        if self._long_meta:
            print(f"[MemoryStore] Rebuilding {len(self._long_meta)} long-term content vectors...")
            contents = [m.get("content", "") for m in self._long_meta]
            vectors = self.embedder.embed_batch(contents)
            if vectors:
                self._long_mat = np.stack([_normalize(v) for v in vectors], axis=1)
            else:
                self._long_mat = np.zeros((0, 0), dtype=np.float32)

        # Rebuild tag vectors
        self._rebuild_tag_vectors()

    def _add_tag_vectors(self, new_tags: list[str]) -> None:
        """Add vectors for new tags (batch operation)."""
        new_tags = [t.strip().lower() for t in new_tags if t and t.strip()]
        new_tags = [t for t in new_tags if t not in self._tag_to_idx]
        if not new_tags:
            return

        print(f"[MemoryStore] Embedding {len(new_tags)} new tags...")
        try:
            vectors = self.embedder.embed_batch(new_tags)
        except Exception as e:
            print(f"[MemoryStore] Warning: Failed to embed tags: {e}")
            return

        for tag, vec in zip(new_tags, vectors):
            idx = len(self._tag_to_idx)
            self._tag_to_idx[tag] = idx
            vec_norm = _normalize(vec)
            if self._tag_vectors.size == 0:
                self._tag_vectors = vec_norm[:, None]
            else:
                self._tag_vectors = np.concatenate([self._tag_vectors, vec_norm[:, None]], axis=1)

    @property
    def dim(self) -> int:
        if self._short_mat.size:
            return int(self._short_mat.shape[0])
        if self._long_mat.size:
            return int(self._long_mat.shape[0])
        if self._tag_vectors.size:
            return int(self._tag_vectors.shape[0])
        return 0

    def stats(self) -> dict[str, Any]:
        return {
            "ok": True,
            "dim": self.dim,
            "short_count": len(self._short_meta),
            "long_count": len(self._long_meta),
            "total_count": len(self._short_meta) + len(self._long_meta),
            "tag_count": len(self._tag_to_idx),
            "model_id": self._stored_model_id,
            "max_memory_strings": self.max_memory_strings,
            "max_short_memory_strings": self.max_short_memory_strings,
            "max_long_memory_strings": self.max_long_memory_strings,
            "short_time_seconds": self.short_time_seconds,
            "long_time_seconds": self.long_time_seconds,
            "data_dir": self.data_dir,
        }

    def store_memory(
        self,
        *,
        content: str,
        tags: Iterable[str] | None = None,
        vector: np.ndarray | None = None,
    ) -> dict[str, Any]:
        content = (content or "").strip()
        if not content:
            raise MemoryError("content is required")

        tag_list = _as_str_list(list(tags) if tags is not None else [])
        # Normalize tags to lowercase
        tag_list = [t.strip().lower() for t in tag_list if t and t.strip()]

        # Vectorize any new tags
        new_tags = [t for t in tag_list if t not in self._tag_to_idx]
        if new_tags:
            self._add_tag_vectors(new_tags)

        # Timestamp is always set by the service at ingest time.
        now = _utc_now_epoch()
        ts_epoch, ts_iso = _parse_timestamp(now)
        self.repartition(now_epoch=now)

        # Embed outside of matrix operations (but still in-process).
        if vector is None:
            vec = _normalize(self.embedder.embed(content))
        else:
            vec = _normalize(np.asarray(vector, dtype=np.float32))

        if self.dim not in (0, int(vec.shape[0])):
            raise MemoryError(
                f"Embedding dimension mismatch: store has dim={self.dim}, got dim={int(vec.shape[0])}. "
                "(Did you change embedding_model?)"
            )

        is_short = (now - float(ts_epoch)) <= float(self.short_time_seconds)

        # Prune if needed BEFORE adding a new item.
        # In per-tier mode this only affects the relevant tier.
        self._enforce_limits(
            now_epoch=now,
            incoming_total=1,
            incoming_short=1 if is_short else 0,
            incoming_long=0 if is_short else 1,
        )

        item: dict[str, Any] = {
            "id": str(uuid.uuid4()),
            "content": content,
            "tags": tag_list,
            "timestamp": ts_iso,
            "timestamp_epoch": float(ts_epoch),
            "usage_count": 0,
        }

        if is_short:
            self._append_short(vec, item)
        else:
            self._append_long(vec, item)

        self._save()
        return item

    def get_top_n_memory(
        self,
        *,
        content: str,
        top_n: int = 2,
        query_vector: np.ndarray | None = None,
    ) -> dict[str, Any]:
        content = (content or "").strip()
        if not content:
            raise MemoryError("content is required")

        if top_n is None:
            top_n = 2
        try:
            top_n_i = int(top_n)
        except Exception as exc:
            raise MemoryError("top_n must be an integer") from exc
        top_n_i = max(1, min(10, top_n_i))

        now = _utc_now_epoch()
        self.repartition(now_epoch=now)
        self._enforce_limits(now_epoch=now)

        if query_vector is None:
            q = _normalize(self.embedder.embed(content))
        else:
            q = _normalize(np.asarray(query_vector, dtype=np.float32))

        short_hits = self._top_n(q, self._short_mat, self._short_meta, top_n_i)
        long_hits = self._top_n(q, self._long_mat, self._long_meta, top_n_i)

        # Increment usage counts for returned items.
        updated = False
        for hit in short_hits:
            idx = hit.get("_index")
            if isinstance(idx, int) and 0 <= idx < len(self._short_meta):
                self._short_meta[idx]["usage_count"] = int(self._short_meta[idx].get("usage_count", 0)) + 1
                updated = True
        for hit in long_hits:
            idx = hit.get("_index")
            if isinstance(idx, int) and 0 <= idx < len(self._long_meta):
                self._long_meta[idx]["usage_count"] = int(self._long_meta[idx].get("usage_count", 0)) + 1
                updated = True

        # Strip internal indices before returning.
        def _clean(h: dict[str, Any]) -> dict[str, Any]:
            h2 = dict(h)
            h2.pop("_index", None)
            return h2

        out = {
            "ok": True,
            "top_n": top_n_i,
            "short_term_memory": [_clean(h) for h in short_hits],
            "long_term_memory": [_clean(h) for h in long_hits],
        }

        if updated:
            self._save()
        return out

    def get_top_n_memory_by_tags(
        self,
        *,
        content: str,
        top_n: int = 3,
        top_k_tags: int = 5,
        query_vector: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Get memories using tag-based pre-filtering for more relevant results.

        Flow:
        1. Vectorize the query
        2. Find top_k_tags most similar tags
        3. Filter memories to only those with matching tags
        4. Run similarity search on filtered set
        5. Return top_n results

        This is more efficient and relevant when there are many memories across
        diverse topics (people, cars, nature, math, etc.)
        """
        content = (content or "").strip()
        if not content:
            raise MemoryError("content is required")

        if top_n is None:
            top_n = 3
        try:
            top_n_i = int(top_n)
        except Exception:
            top_n_i = 3
        top_n_i = max(1, min(10, top_n_i))

        if top_k_tags is None:
            top_k_tags = 5
        try:
            top_k_tags_i = int(top_k_tags)
        except Exception:
            top_k_tags_i = 5
        top_k_tags_i = max(1, min(20, top_k_tags_i))

        now = _utc_now_epoch()
        self.repartition(now_epoch=now)
        self._enforce_limits(now_epoch=now)

        # Embed the query
        if query_vector is None:
            q = _normalize(self.embedder.embed(content))
        else:
            q = _normalize(np.asarray(query_vector, dtype=np.float32))

        # Step 1: Find top_k most similar tags
        matched_tags: list[str] = []
        tag_scores: dict[str, float] = {}
        
        if self._tag_vectors.size > 0 and self._tag_to_idx:
            # Compute similarity with all tag vectors
            tag_sims = (self._tag_vectors.T @ q).astype(np.float32, copy=False)
            
            # Get top_k tag indices
            k = min(top_k_tags_i, len(self._tag_to_idx))
            top_tag_indices = np.argsort(-tag_sims)[:k]
            
            # Map indices back to tag names
            idx_to_tag = {v: k for k, v in self._tag_to_idx.items()}
            for idx in top_tag_indices.tolist():
                tag_name = idx_to_tag.get(idx)
                if tag_name:
                    matched_tags.append(tag_name)
                    tag_scores[tag_name] = float(tag_sims[idx])

        # Step 2: Filter memories by matched tags
        def _has_matching_tag(meta: dict[str, Any]) -> bool:
            if not matched_tags:
                return True  # No tag filtering if no tags found
            mem_tags = [t.lower() for t in meta.get("tags", []) if t]
            return any(t in matched_tags for t in mem_tags)

        # Step 3: Run similarity search on filtered sets
        short_hits = self._top_n_filtered(q, self._short_mat, self._short_meta, top_n_i, _has_matching_tag)
        long_hits = self._top_n_filtered(q, self._long_mat, self._long_meta, top_n_i, _has_matching_tag)

        # Increment usage counts
        updated = False
        for hit in short_hits:
            idx = hit.get("_index")
            if isinstance(idx, int) and 0 <= idx < len(self._short_meta):
                self._short_meta[idx]["usage_count"] = int(self._short_meta[idx].get("usage_count", 0)) + 1
                updated = True
        for hit in long_hits:
            idx = hit.get("_index")
            if isinstance(idx, int) and 0 <= idx < len(self._long_meta):
                self._long_meta[idx]["usage_count"] = int(self._long_meta[idx].get("usage_count", 0)) + 1
                updated = True

        def _clean(h: dict[str, Any]) -> dict[str, Any]:
            h2 = dict(h)
            h2.pop("_index", None)
            return h2

        out = {
            "ok": True,
            "top_n": top_n_i,
            "matched_tags": matched_tags,
            "tag_scores": tag_scores,
            "short_term_memory": [_clean(h) for h in short_hits],
            "long_term_memory": [_clean(h) for h in long_hits],
        }

        if updated:
            self._save()
        return out

    def _top_n_filtered(
        self,
        q: np.ndarray,
        mat: np.ndarray,
        meta: list[dict[str, Any]],
        n: int,
        filter_fn: Any,
    ) -> list[dict[str, Any]]:
        """Get top N memories that pass the filter function."""
        if n <= 0 or mat.size == 0 or not meta:
            return []
        q = np.asarray(q, dtype=np.float32).reshape(-1)
        
        # Find indices that pass filter
        valid_indices = [i for i, m in enumerate(meta) if filter_fn(m)]
        if not valid_indices:
            return []
        
        # Extract only valid columns for similarity
        valid_mat = mat[:, valid_indices]
        sims = (valid_mat.T @ q).astype(np.float32, copy=False)
        
        k = min(int(n), len(valid_indices))
        sorted_local_indices = np.argsort(-sims)[:k]
        
        out: list[dict[str, Any]] = []
        for local_idx in sorted_local_indices.tolist():
            original_idx = valid_indices[local_idx]
            m = dict(meta[original_idx])
            m["score"] = float(sims[local_idx])
            m["_index"] = original_idx
            out.append(m)
        return out

    def repartition(self, *, now_epoch: float) -> None:
        """Move items between short and long based on current time."""
        now = float(now_epoch)

        # Optional age-based prune.
        if self.prune_older_than_long_time and self.long_time_seconds > 0:
            self._prune_older_than(now)

        # Move short -> long
        if self._short_meta:
            ages = np.asarray([now - float(m.get("timestamp_epoch", now)) for m in self._short_meta], dtype=np.float64)
            move_mask = ages > float(self.short_time_seconds)
            if bool(np.any(move_mask)):
                self._move_short_to_long(move_mask)

        # Move long -> short (rare, but handles clock skew)
        if self._long_meta:
            ages = np.asarray([now - float(m.get("timestamp_epoch", now)) for m in self._long_meta], dtype=np.float64)
            move_mask = ages <= float(self.short_time_seconds)
            if bool(np.any(move_mask)):
                self._move_long_to_short(move_mask)

    def _prune_older_than(self, now_epoch: float) -> None:
        cutoff = float(now_epoch) - float(self.long_time_seconds)
        if cutoff <= 0:
            return

        # Short
        if self._short_meta:
            keep = [float(m.get("timestamp_epoch", now_epoch)) >= cutoff for m in self._short_meta]
            if not all(keep):
                self._short_meta, self._short_mat = self._filter_by_keep(self._short_meta, self._short_mat, keep)

        # Long
        if self._long_meta:
            keep = [float(m.get("timestamp_epoch", now_epoch)) >= cutoff for m in self._long_meta]
            if not all(keep):
                self._long_meta, self._long_mat = self._filter_by_keep(self._long_meta, self._long_mat, keep)

    @staticmethod
    def _filter_by_keep(meta: list[dict[str, Any]], mat: np.ndarray, keep: list[bool]) -> tuple[list[dict[str, Any]], np.ndarray]:
        idx = [i for i, k in enumerate(keep) if k]
        new_meta = [meta[i] for i in idx]
        if mat.size == 0:
            return new_meta, mat
        if not idx:
            return [], np.zeros((mat.shape[0], 0), dtype=np.float32)
        new_mat = mat[:, idx]
        return new_meta, np.asarray(new_mat, dtype=np.float32)

    def _move_short_to_long(self, move_mask: np.ndarray) -> None:
        idx_move = np.where(move_mask)[0].tolist()
        idx_keep = np.where(~move_mask)[0].tolist()

        moved_meta = [self._short_meta[i] for i in idx_move]
        kept_meta = [self._short_meta[i] for i in idx_keep]

        moved_mat = self._short_mat[:, idx_move] if self._short_mat.size else np.zeros((self.dim, 0), dtype=np.float32)
        kept_mat = self._short_mat[:, idx_keep] if self._short_mat.size else np.zeros((self.dim, 0), dtype=np.float32)

        self._short_meta = kept_meta
        self._short_mat = np.asarray(kept_mat, dtype=np.float32)

        for i, m in enumerate(moved_meta):
            col = moved_mat[:, i] if moved_mat.size else None
            if col is None:
                continue
            self._append_long(col, m)

    def _move_long_to_short(self, move_mask: np.ndarray) -> None:
        idx_move = np.where(move_mask)[0].tolist()
        idx_keep = np.where(~move_mask)[0].tolist()

        moved_meta = [self._long_meta[i] for i in idx_move]
        kept_meta = [self._long_meta[i] for i in idx_keep]

        moved_mat = self._long_mat[:, idx_move] if self._long_mat.size else np.zeros((self.dim, 0), dtype=np.float32)
        kept_mat = self._long_mat[:, idx_keep] if self._long_mat.size else np.zeros((self.dim, 0), dtype=np.float32)

        self._long_meta = kept_meta
        self._long_mat = np.asarray(kept_mat, dtype=np.float32)

        for i, m in enumerate(moved_meta):
            col = moved_mat[:, i] if moved_mat.size else None
            if col is None:
                continue
            self._append_short(col, m)

    def _append_short(self, vec: np.ndarray, meta: dict[str, Any]) -> None:
        vec = np.asarray(vec, dtype=np.float32).reshape(-1)
        if self._short_mat.size == 0:
            self._short_mat = vec[:, None]
        else:
            self._short_mat = np.concatenate([self._short_mat, vec[:, None]], axis=1)
        self._short_meta.append(meta)

    def _append_long(self, vec: np.ndarray, meta: dict[str, Any]) -> None:
        vec = np.asarray(vec, dtype=np.float32).reshape(-1)
        if self._long_mat.size == 0:
            self._long_mat = vec[:, None]
        else:
            self._long_mat = np.concatenate([self._long_mat, vec[:, None]], axis=1)
        self._long_meta.append(meta)

    @staticmethod
    def _top_n(q: np.ndarray, mat: np.ndarray, meta: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
        if n <= 0 or mat.size == 0 or not meta:
            return []
        q = np.asarray(q, dtype=np.float32).reshape(-1)
        # cosine similarity because all vectors are unit-normalized.
        sims = (mat.T @ q).astype(np.float32, copy=False)  # (n_items,)
        k = min(int(n), int(sims.shape[0]))
        # argsort is fine for n<=10.
        idx = np.argsort(-sims)[:k]
        out: list[dict[str, Any]] = []
        for i in idx.tolist():
            m = dict(meta[int(i)])
            m["score"] = float(sims[int(i)])
            m["_index"] = int(i)
            out.append(m)
        return out

    def _enforce_limits(
        self,
        *,
        now_epoch: float,
        incoming_total: int = 0,
        incoming_short: int = 0,
        incoming_long: int = 0,
    ) -> None:
        """Enforce capacity limits.

        - If per-tier limits are configured, prune each tier independently.
        - Otherwise, fall back to legacy global pruning across both tiers.
        """

        per_tier = (self.max_short_memory_strings is not None) or (self.max_long_memory_strings is not None)
        if per_tier:
            if self.max_short_memory_strings is not None and int(self.max_short_memory_strings) > 0:
                target_s = int(self.max_short_memory_strings)
                while len(self._short_meta) + int(incoming_short) > target_s and len(self._short_meta) > 0:
                    self._prune_one_in_bucket(bucket="short", now_epoch=now_epoch)
            if self.max_long_memory_strings is not None and int(self.max_long_memory_strings) > 0:
                target_l = int(self.max_long_memory_strings)
                while len(self._long_meta) + int(incoming_long) > target_l and len(self._long_meta) > 0:
                    self._prune_one_in_bucket(bucket="long", now_epoch=now_epoch)
            return

        # Legacy global mode
        if self.max_memory_strings is None or int(self.max_memory_strings) <= 0:
            return
        total = len(self._short_meta) + len(self._long_meta)
        target = int(self.max_memory_strings)
        while total + int(incoming_total) > target and total > 0:
            self._prune_one_global(now_epoch=now_epoch)
            total = len(self._short_meta) + len(self._long_meta)

    def _prune_one_in_bucket(self, *, bucket: str, now_epoch: float) -> None:
        if bucket == "short":
            meta = self._short_meta
            mat = self._short_mat
        elif bucket == "long":
            meta = self._long_meta
            mat = self._long_mat
        else:
            raise MemoryError(f"Invalid bucket: {bucket!r}")

        if not meta:
            return

        best: tuple[int, float, int] | None = None
        # tuple: (usage_count, timestamp_epoch, index)
        for i, m in enumerate(meta):
            u = int(m.get("usage_count", 0))
            ts = float(m.get("timestamp_epoch", now_epoch))
            cand = (u, ts, i)
            if best is None or cand < best:
                best = cand
        if best is None:
            return
        _, _, idx = best

        meta.pop(idx)
        if mat.size:
            mat = np.delete(mat, idx, axis=1)
            if mat.shape[1] == 0:
                mat = np.zeros((self.dim, 0), dtype=np.float32)
        if bucket == "short":
            self._short_mat = np.asarray(mat, dtype=np.float32)
        else:
            self._long_mat = np.asarray(mat, dtype=np.float32)

    def _prune_one_global(self, *, now_epoch: float) -> None:
        # Find least-used across both.
        best: tuple[int, float, str, int] | None = None
        # tuple: (usage_count, timestamp_epoch, bucket, index)

        for bucket, meta in (("short", self._short_meta), ("long", self._long_meta)):
            for i, m in enumerate(meta):
                u = int(m.get("usage_count", 0))
                ts = float(m.get("timestamp_epoch", now_epoch))
                cand = (u, ts, bucket, i)
                if best is None or cand < best:
                    best = cand

        if best is None:
            return
        _, _, bucket, idx = best

        if bucket == "short":
            self._short_meta.pop(idx)
            if self._short_mat.size:
                self._short_mat = np.delete(self._short_mat, idx, axis=1)
                if self._short_mat.shape[1] == 0:
                    self._short_mat = np.zeros((self.dim, 0), dtype=np.float32)
        else:
            self._long_meta.pop(idx)
            if self._long_mat.size:
                self._long_mat = np.delete(self._long_mat, idx, axis=1)
                if self._long_mat.shape[1] == 0:
                    self._long_mat = np.zeros((self.dim, 0), dtype=np.float32)

    def _paths(self) -> dict[str, str]:
        return {
            "short_mat": os.path.join(self.data_dir, "short_vectors.npy"),
            "short_meta": os.path.join(self.data_dir, "short_meta.json"),
            "long_mat": os.path.join(self.data_dir, "long_vectors.npy"),
            "long_meta": os.path.join(self.data_dir, "long_meta.json"),
            # Tag system files
            "tag_vectors": os.path.join(self.data_dir, "tag_vectors.npy"),
            "tag_meta": os.path.join(self.data_dir, "tag_meta.json"),
        }

    def _load(self) -> None:
        _ensure_dir(self.data_dir)
        p = self._paths()

        def _load_json(path: str) -> list[dict[str, Any]]:
            if not os.path.exists(path):
                return []
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if not isinstance(obj, list):
                return []
            out: list[dict[str, Any]] = []
            for x in obj:
                if isinstance(x, dict):
                    out.append(x)
            return out

        def _load_json_dict(path: str) -> dict[str, Any]:
            if not os.path.exists(path):
                return {}
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if not isinstance(obj, dict):
                return {}
            return obj

        def _load_mat(path: str) -> np.ndarray:
            if not os.path.exists(path):
                return np.zeros((0, 0), dtype=np.float32)
            arr = np.load(path)
            if not isinstance(arr, np.ndarray):
                return np.zeros((0, 0), dtype=np.float32)
            if arr.ndim != 2:
                return np.zeros((0, 0), dtype=np.float32)
            return np.asarray(arr, dtype=np.float32)

        self._short_meta = _load_json(p["short_meta"])
        self._long_meta = _load_json(p["long_meta"])
        self._short_mat = _load_mat(p["short_mat"])
        self._long_mat = _load_mat(p["long_mat"])

        # Load tag system
        tag_meta = _load_json_dict(p["tag_meta"])
        self._tag_to_idx = {str(k): int(v) for k, v in tag_meta.get("tag_to_idx", {}).items()}
        self._stored_model_id = tag_meta.get("model_id")
        self._tag_vectors = _load_mat(p["tag_vectors"])

        # Reconcile tag vector size
        if self._tag_vectors.size and len(self._tag_to_idx) != self._tag_vectors.shape[1]:
            print(f"[MemoryStore] Tag vector mismatch: {len(self._tag_to_idx)} tags vs {self._tag_vectors.shape[1]} vectors")
            # Reset and rebuild
            self._tag_to_idx = {}
            self._tag_vectors = np.zeros((0, 0), dtype=np.float32)

        # Reconcile sizes.
        if self._short_mat.size and len(self._short_meta) != self._short_mat.shape[1]:
            n = min(len(self._short_meta), int(self._short_mat.shape[1]))
            self._short_meta = self._short_meta[:n]
            self._short_mat = self._short_mat[:, :n]

        if self._long_mat.size and len(self._long_meta) != self._long_mat.shape[1]:
            n = min(len(self._long_meta), int(self._long_mat.shape[1]))
            self._long_meta = self._long_meta[:n]
            self._long_mat = self._long_mat[:, :n]

        # Ensure required keys exist.
        def _fix_meta(meta: list[dict[str, Any]]) -> None:
            for m in meta:
                if "id" not in m:
                    m["id"] = str(uuid.uuid4())
                if "content" not in m:
                    m["content"] = ""
                if "tags" not in m or not isinstance(m.get("tags"), list):
                    m["tags"] = _as_str_list(m.get("tags"))
                # Normalize tags to lowercase
                m["tags"] = [t.strip().lower() for t in m["tags"] if t and t.strip()]
                if "usage_count" not in m:
                    m["usage_count"] = 0
                if "timestamp_epoch" not in m:
                    try:
                        ts_epoch, ts_iso = _parse_timestamp(m.get("timestamp"))
                    except Exception:
                        ts_epoch, ts_iso = _parse_timestamp(None)
                    m["timestamp_epoch"] = float(ts_epoch)
                    m["timestamp"] = ts_iso
                if "timestamp" not in m:
                    dt = datetime.fromtimestamp(float(m.get("timestamp_epoch", _utc_now_epoch())), tz=timezone.utc)
                    m["timestamp"] = _dt_to_iso_z(dt)

        _fix_meta(self._short_meta)
        _fix_meta(self._long_meta)

        # Ensure normalization (best-effort).
        if self._short_mat.size:
            self._short_mat = self._renormalize_mat(self._short_mat)
        if self._long_mat.size:
            self._long_mat = self._renormalize_mat(self._long_mat)
        if self._tag_vectors.size:
            self._tag_vectors = self._renormalize_mat(self._tag_vectors)

    @staticmethod
    def _renormalize_mat(mat: np.ndarray) -> np.ndarray:
        mat = np.asarray(mat, dtype=np.float32)
        if mat.size == 0:
            return mat
        norms = np.linalg.norm(mat, axis=0)
        norms = np.where(norms == 0, 1.0, norms).astype(np.float32)
        return mat / norms

    def _save(self) -> None:
        _ensure_dir(self.data_dir)
        p = self._paths()

        _atomic_write_text(p["short_meta"], json.dumps(self._short_meta, ensure_ascii=False, indent=2))
        _atomic_write_text(p["long_meta"], json.dumps(self._long_meta, ensure_ascii=False, indent=2))

        # Save tag metadata
        tag_meta = {
            "model_id": self._stored_model_id,
            "tag_to_idx": self._tag_to_idx,
        }
        _atomic_write_text(p["tag_meta"], json.dumps(tag_meta, ensure_ascii=False, indent=2))

        # NPY atomic save: write to bytes then atomic write.
        def _save_mat(path: str, mat: np.ndarray) -> None:
            mat = np.asarray(mat, dtype=np.float32)
            # Guarantee shape (dim, n)
            if mat.ndim != 2:
                mat = np.zeros((0, 0), dtype=np.float32)
            import io

            bio = io.BytesIO()
            np.save(bio, mat)
            _atomic_write_bytes(path, bio.getvalue())

        _save_mat(p["short_mat"], self._short_mat)
        _save_mat(p["long_mat"], self._long_mat)
        _save_mat(p["tag_vectors"], self._tag_vectors)

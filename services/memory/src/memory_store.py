from __future__ import annotations

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


class MemoryStore:
    """Persistent two-tier embedding store (short-term + long-term)."""

    def __init__(
        self,
        *,
        data_dir: str,
        embedder: OpenAIEmbedder,
        short_time_seconds: float = 3600.0,
        long_time_seconds: float = 2592000.0,
        prune_older_than_long_time: bool = False,
        max_memory_strings: int = 1000,
    ):
        self.data_dir = os.path.abspath(data_dir)
        self.embedder = embedder
        self.short_time_seconds = float(short_time_seconds)
        self.long_time_seconds = float(long_time_seconds)
        self.prune_older_than_long_time = bool(prune_older_than_long_time)
        self.max_memory_strings = int(max_memory_strings)

        self._short_meta: list[dict[str, Any]] = []
        self._long_meta: list[dict[str, Any]] = []

        # Matrices are shape (dim, n) where each column is a unit vector.
        self._short_mat: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self._long_mat: np.ndarray = np.zeros((0, 0), dtype=np.float32)

        self._load()
        # Repartition on startup (short memories may have aged into long).
        self.repartition(now_epoch=_utc_now_epoch())
        self._enforce_limits(now_epoch=_utc_now_epoch())
        self._save()

    @property
    def dim(self) -> int:
        if self._short_mat.size:
            return int(self._short_mat.shape[0])
        if self._long_mat.size:
            return int(self._long_mat.shape[0])
        return 0

    def stats(self) -> dict[str, Any]:
        return {
            "ok": True,
            "dim": self.dim,
            "short_count": len(self._short_meta),
            "long_count": len(self._long_meta),
            "total_count": len(self._short_meta) + len(self._long_meta),
            "max_memory_strings": self.max_memory_strings,
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

        # Prune if needed BEFORE adding a new item.
        self._enforce_limits(now_epoch=now, incoming=1)

        item: dict[str, Any] = {
            "id": str(uuid.uuid4()),
            "content": content,
            "tags": tag_list,
            "timestamp": ts_iso,
            "timestamp_epoch": float(ts_epoch),
            "usage_count": 0,
        }

        is_short = (now - float(ts_epoch)) <= float(self.short_time_seconds)
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

    def _enforce_limits(self, *, now_epoch: float, incoming: int = 0) -> None:
        if self.max_memory_strings <= 0:
            return
        total = len(self._short_meta) + len(self._long_meta)
        target = int(self.max_memory_strings)
        while total + int(incoming) > target and total > 0:
            self._prune_one(now_epoch=now_epoch)
            total = len(self._short_meta) + len(self._long_meta)

    def _prune_one(self, *, now_epoch: float) -> None:
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

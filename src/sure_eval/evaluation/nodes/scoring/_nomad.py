"""Official NOMAD non-matching-reference speech quality provider."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


class NOMADProvider:
    """Score speech by mean NOMAD distance to clean non-matching references.

    NOMAD is an embedding distance, not a MOS predictor. The official package
    expects its two checkpoints at fixed relative paths and downloads them at
    import time. This provider stages explicit, validated assets in a private
    temporary directory before loading the official implementation, so a
    missing asset fails instead of triggering an implicit network download.
    """

    MODEL_FILENAME = "nomad_best_model.pt"
    WAV2VEC_FILENAME = "wav2vec_small.pt"
    EMBEDDING_DIMENSION = 256
    AUDIO_SUFFIXES = {".wav", ".flac", ".ogg", ".mp3", ".m4a"}

    def __init__(
        self,
        *,
        checkpoint: str | None = None,
        wav2vec_checkpoint: str | None = None,
        non_matching_reference_dir: str | None = None,
        reference_dir: str | None = None,
        device: str = "cpu",
    ) -> None:
        node_dir = Path(__file__).resolve().parent / "nomad"
        self.checkpoint = Path(
            checkpoint
            or os.environ.get("SURE_SE_NOMAD_CHECKPOINT")
            or node_dir / "checkpoints" / self.MODEL_FILENAME
        ).expanduser()
        self.wav2vec_checkpoint = Path(
            wav2vec_checkpoint
            or os.environ.get("SURE_SE_NOMAD_WAV2VEC_CHECKPOINT")
            or node_dir / "checkpoints" / self.WAV2VEC_FILENAME
        ).expanduser()
        reference_value = (
            non_matching_reference_dir
            or reference_dir
            or os.environ.get("SURE_SE_NOMAD_REFERENCE_DIR")
            or os.environ.get("SURE_SE_NOMAD_NMR_DIR")
        )
        self.non_matching_reference_dir = (
            Path(reference_value).expanduser() if reference_value else None
        )
        self.device = device
        self._engine: Any | None = None
        self._reference_paths: tuple[Path, ...] | None = None
        self._reference_embeddings: np.ndarray | None = None
        self._embedding_cache: dict[str, np.ndarray] = {}
        self._scores: dict[str, float] = {}
        self._asset_hashes: tuple[str, str] | None = None
        self._runtime_dir: tempfile.TemporaryDirectory[str] | None = None

    def _validate_assets(self) -> tuple[Path, Path, tuple[Path, ...]]:
        checkpoint = self.checkpoint.resolve()
        wav2vec = self.wav2vec_checkpoint.resolve()
        if not checkpoint.is_file():
            raise RuntimeError(
                "NOMAD requires SURE_SE_NOMAD_CHECKPOINT pointing to "
                f"{self.MODEL_FILENAME}: {checkpoint}"
            )
        if not wav2vec.is_file():
            raise RuntimeError(
                "NOMAD requires SURE_SE_NOMAD_WAV2VEC_CHECKPOINT pointing to "
                f"{self.WAV2VEC_FILENAME}: {wav2vec}"
            )
        if self.non_matching_reference_dir is None or not self.non_matching_reference_dir.is_dir():
            raise RuntimeError(
                "NOMAD requires SURE_SE_NOMAD_REFERENCE_DIR pointing to a directory "
                "of clean non-matching reference audio"
            )
        references = tuple(
            sorted(
                path
                for path in self.non_matching_reference_dir.iterdir()
                if path.is_file() and path.suffix.lower() in self.AUDIO_SUFFIXES
            )
        )
        if not references:
            raise RuntimeError(
                f"NOMAD non-matching reference directory has no supported audio: "
                f"{self.non_matching_reference_dir}"
            )
        return checkpoint, wav2vec, references

    def _load_engine(self) -> Any:
        if self._engine is not None:
            return self._engine
        checkpoint, wav2vec, _references = self._validate_assets()
        try:
            distribution = importlib.metadata.distribution("nomad_audio")
            source = Path(distribution.locate_file("nomad_audio/nomad.py"))
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                "NOMAD requires the official `nomad_audio==0.0.8` package"
            ) from exc
        if not source.is_file():
            raise RuntimeError(f"NOMAD package source is missing: {source}")

        # The released package uses relative pt-models paths and downloads on
        # import. Symlinks in a private temporary cwd keep that behavior
        # deterministic while still loading the caller-selected assets.
        self._runtime_dir = tempfile.TemporaryDirectory(prefix="sure_nomad_", dir="/tmp")
        models_dir = Path(self._runtime_dir.name) / "pt-models"
        models_dir.mkdir()
        os.symlink(wav2vec, models_dir / self.WAV2VEC_FILENAME)
        os.symlink(checkpoint, models_dir / self.MODEL_FILENAME)
        module_name = f"_sure_nomad_official_{id(self)}"
        spec = importlib.util.spec_from_file_location(module_name, source)
        if spec is None or spec.loader is None:
            raise RuntimeError("could not load the official NOMAD implementation")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        previous_cwd = Path.cwd()
        try:
            os.chdir(self._runtime_dir.name)
            spec.loader.exec_module(module)
            self._engine = module.Nomad(device=self.device)
        except Exception as exc:
            raise RuntimeError(f"failed to load the official NOMAD model: {exc}") from exc
        finally:
            os.chdir(previous_cwd)
        return self._engine

    def _embedding(self, path: Path) -> np.ndarray:
        key = str(path.resolve())
        cached = self._embedding_cache.get(key)
        if cached is not None:
            return cached
        engine = self._load_engine()
        try:
            import torch

            with torch.inference_mode():
                waveform = engine.load_processing(key, trim=False).to(engine.DEVICE)
                embedding = engine.model(waveform, None).reshape(-1).detach().cpu().numpy()
        except Exception as exc:
            raise RuntimeError(f"NOMAD failed to embed audio: {path}") from exc
        embedding = np.asarray(embedding, dtype=np.float32)
        if embedding.shape != (self.EMBEDDING_DIMENSION,) or not np.isfinite(embedding).all():
            raise ValueError(f"NOMAD returned an invalid embedding for {path}")
        self._embedding_cache[key] = embedding
        return embedding

    def _reference_matrix(self) -> np.ndarray:
        if self._reference_embeddings is None:
            _checkpoint, _wav2vec, references = self._validate_assets()
            self._reference_paths = references
            self._reference_embeddings = np.stack(
                [self._embedding(path) for path in references], axis=0
            )
        return self._reference_embeddings

    def __call__(self, prediction: str, reference: str = "", **kwargs: Any) -> dict[str, Any]:
        del reference, kwargs
        key = str(Path(prediction).resolve())
        score = self._scores.get(key)
        if score is None:
            candidate = self._embedding(Path(prediction))
            references = self._reference_matrix()
            distances = np.linalg.norm(references - candidate[None, :], axis=1)
            score = float(np.mean(distances))
            if not math.isfinite(score):
                raise ValueError("NOMAD returned a non-finite distance")
            self._scores[key] = score
        checkpoint, wav2vec, references = self._validate_assets()
        if self._asset_hashes is None:
            self._asset_hashes = (self._sha256(checkpoint), self._sha256(wav2vec))
        return {
            "nomad": score,
            "score": score,
            "backend": "nomad-audio-0.0.8",
            "higher_is_better": False,
            "num_non_matching_references": len(references),
            "checkpoint_sha256": self._asset_hashes[0],
            "wav2vec_sha256": self._asset_hashes[1],
        }

    def score_batch(
        self, rows: list[tuple[str, str]], *, metric_name: str
    ) -> list[dict[str, Any]]:
        if metric_name != "nomad":
            raise ValueError(f"NOMAD provider received unsupported metric: {metric_name}")
        return [self(prediction) for _sample_id, prediction in rows]

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()


__all__ = ["NOMADProvider"]

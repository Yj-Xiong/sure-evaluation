"""Route identity and score-definition checks for opt-in SE metrics."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import sys
import types
import wave
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from sure_eval.cli import app
from sure_eval.evaluation.cli_adapters import build_pipeline_spec, list_metric_routes
from sure_eval.evaluation.nodes.scoring._score_components import ScoreComponentProvider
from sure_eval.evaluation.nodes.scoring._full_reference_audio import (
    ESTOIProvider, LSDProvider,
)
from sure_eval.evaluation.nodes.scoring.lps.node import LPSProvider, levenshtein


NEW_METRICS = (
    "estoi", "lsd", "nsrmr", "lps", "visqol", "nomad", "dnsmos_p808", "dnsmos_sig",
    "dnsmos_bak", "squim_stoi", "squim_pesq", "squim_si_sdr",
    "squim_mos", "scoreq",
    "sq_ast_mos", "sq_ast_noi", "sq_ast_dis", "sq_ast_col", "sq_ast_loud",
    "nisqa_mos", "nisqa_noi", "nisqa_dis", "nisqa_col", "nisqa_loud",
)


def _wav(path: Path, signal: np.ndarray, sample_rate: int = 16000) -> None:
    samples = (np.clip(signal, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(samples.tobytes())


def test_new_atomic_routes_and_original_default_bundle() -> None:
    for metric in NEW_METRICS:
        routes = list_metric_routes("se", metric=metric)
        assert routes["count"] == 1
        route = routes["routes"][0]
        spec = build_pipeline_spec("se", pipeline_id=route["pipeline_id"])
        assert spec["pipeline_id"] == route["pipeline_id"]
        assert spec["metric"] == metric
        assert spec["computation_node_ids"] == route["computation_node_ids"]
        assert route["default"] is True
    default = build_pipeline_spec("se")
    assert default["metrics"] == ["si-sdr", "stoi", "pesq", "dnsmos", "wv-mos", "utmos"]
    assert default["pipeline_id"].endswith("utmos.utmos_v1")


def test_visqol_is_a_full_reference_route() -> None:
    route = list_metric_routes("se", metric="visqol")["routes"][0]
    assert route["pipeline_id"] == "se.any.visqol.visqol_v1"
    assert route["environments"][0]["group"] == "se-full-reference"
    assert route["input_contract"] == "scoring/visqol"

    from sure_eval.evaluation.tasks.se.pipeline import evaluate_se_samples
    from sure_eval.evaluation.tasks.se.types import SESample

    report = evaluate_se_samples(
        [SESample(enhanced_audio="enhanced.wav", reference_audio="clean.wav", sample_id="utt1")],
        metrics=("visqol",),
        reference_providers={"visqol": lambda prediction, reference, **kwargs: {"visqol": 3.7}},
    )
    assert report.pipeline_id == "se.any.visqol.visqol_v1"
    assert report.input_contract.required_roles == ("enhanced_audio", "reference_audio")
    assert report.details["results"]["visqol"]["score"] == pytest.approx(3.7)


def test_visqol_provider_calls_google_cli_with_paired_16k_audio(
    monkeypatch, tmp_path: Path
) -> None:
    from sure_eval.evaluation.nodes.scoring._full_reference_audio import ViSQOLProvider

    model = tmp_path / ViSQOLProvider.MODEL_FILENAME
    model.write_bytes(b"test model")
    binary = tmp_path / "visqol"
    binary.write_bytes(b"test binary")
    binary.chmod(0o755)
    reference = tmp_path / "reference.wav"
    enhanced = tmp_path / "enhanced.wav"
    _wav(reference, np.array([0.0, 0.5, -0.5, 0.0], dtype=np.float32))
    _wav(enhanced, np.array([0.0, 0.25, -0.25, 0.0], dtype=np.float32))

    observed = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        assert kwargs["check"] is False
        flags = dict(arg.split("=", 1) for arg in command[1:])
        assert flags["--use_speech_mode"] == "true"
        assert flags["--use_lattice_model"] == "true"
        assert flags["--use_unscaled_speech_mos_mapping"] == "false"
        assert flags["--search_window_radius"] == "60"
        assert flags["--similarity_to_quality_model"] == str(model)
        with wave.open(flags["--reference_file"], "rb") as clean:
            assert (
                clean.getframerate(), clean.getnchannels(), clean.getsampwidth()
            ) == (16000, 1, 2)
            observed["clean"] = np.frombuffer(clean.readframes(4), dtype="<i2")
        with wave.open(flags["--degraded_file"], "rb") as degraded:
            assert (
                degraded.getframerate(), degraded.getnchannels(), degraded.getsampwidth()
            ) == (16000, 1, 2)
            observed["degraded"] = np.frombuffer(degraded.readframes(4), dtype="<i2")
        with open(flags["--results_csv"], "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("reference", "degraded", "moslqo"))
            writer.writerow((flags["--reference_file"], flags["--degraded_file"], 3.25))
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = ViSQOLProvider(binary_path=str(binary), model_path=str(model))(
        str(enhanced), str(reference)
    )
    assert observed["command"][0] == str(binary)
    assert observed["clean"][1] > observed["degraded"][1]
    assert result["visqol"] == pytest.approx(3.25)
    assert result["backend"] == "google-visqol-v3.3.3-cli"
    assert result["binary_sha256"] == hashlib.sha256(b"test binary").hexdigest()
    assert result["model_sha256"] == hashlib.sha256(b"test model").hexdigest()


def test_visqol_provider_rejects_missing_official_model(monkeypatch) -> None:
    from sure_eval.evaluation.nodes.scoring._full_reference_audio import ViSQOLProvider

    monkeypatch.setenv("SURE_SE_VISQOL_BINARY", "/no/such/visqol")
    monkeypatch.delenv("SURE_SE_VISQOL_LATTICE_MODEL", raising=False)
    with pytest.raises(RuntimeError, match="SURE_SE_VISQOL_LATTICE_MODEL"):
        ViSQOLProvider()._validate_assets()


def test_visqol_provider_rejects_cli_success_without_score(monkeypatch, tmp_path: Path) -> None:
    from sure_eval.evaluation.nodes.scoring._full_reference_audio import ViSQOLProvider

    binary = tmp_path / "visqol"
    binary.write_bytes(b"test binary")
    binary.chmod(0o755)
    model = tmp_path / ViSQOLProvider.MODEL_FILENAME
    model.write_bytes(b"test model")
    audio = tmp_path / "audio.wav"
    _wav(audio, np.array([0.0, 0.5, 0.0], dtype=np.float32))
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: types.SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    provider = ViSQOLProvider(binary_path=str(binary), model_path=str(model))
    with pytest.raises(RuntimeError, match="no results CSV"):
        provider(str(audio), str(audio))


def test_nomad_route_uses_non_matching_reference_protocol() -> None:
    route = list_metric_routes("se", metric="nomad")["routes"][0]
    assert route["pipeline_id"] == "se.any.nomad.nomad_v1"
    assert route["input_contract"] == "scoring/nomad"
    assert route["environments"][0]["group"] == "se-no-reference"

    from sure_eval.evaluation.tasks.se.pipeline import evaluate_se_samples
    from sure_eval.evaluation.tasks.se.types import SESample

    report = evaluate_se_samples(
        [SESample(enhanced_audio="enhanced.wav", sample_id="utt1")],
        metrics=("nomad",),
        mos_providers={"nomad": lambda prediction, reference, **kwargs: {"nomad": 0.42}},
    )
    result = report.details["results"]["nomad"]
    assert report.pipeline_id == "se.any.nomad.nomad_v1"
    assert result["score"] == pytest.approx(0.42)
    assert result["higher_is_better"] is False
    assert result["unit"] == "embedding_distance"


def test_nomad_provider_averages_fixed_clean_reference_pool(monkeypatch, tmp_path: Path) -> None:
    from sure_eval.evaluation.nodes.scoring._nomad import NOMADProvider

    checkpoint = tmp_path / "nomad_best_model.pt"
    wav2vec = tmp_path / "wav2vec_small.pt"
    checkpoint.write_bytes(b"checkpoint")
    wav2vec.write_bytes(b"wav2vec")
    reference_dir = tmp_path / "reference_pool"
    reference_dir.mkdir()
    first = reference_dir / "first.wav"
    second = reference_dir / "second.wav"
    prediction = tmp_path / "prediction.wav"
    for path in (first, second, prediction):
        path.write_bytes(b"audio")

    provider = NOMADProvider(
        checkpoint=str(checkpoint),
        wav2vec_checkpoint=str(wav2vec),
        reference_dir=str(reference_dir),
    )
    values = {
        str(first.resolve()): np.array([0.0, 0.0], dtype=np.float32),
        str(second.resolve()): np.array([2.0, 0.0], dtype=np.float32),
        str(prediction.resolve()): np.array([1.0, 0.0], dtype=np.float32),
    }
    monkeypatch.setattr(provider, "_embedding", lambda path: values[str(path.resolve())])
    result = provider(str(prediction))
    assert result["nomad"] == pytest.approx(1.0)
    assert result["score"] == pytest.approx(1.0)
    assert result["num_non_matching_references"] == 2
    assert result["higher_is_better"] is False


def test_nomad_provider_requires_explicit_reference_pool(monkeypatch, tmp_path: Path) -> None:
    from sure_eval.evaluation.nodes.scoring._nomad import NOMADProvider

    checkpoint = tmp_path / "nomad_best_model.pt"
    wav2vec = tmp_path / "wav2vec_small.pt"
    checkpoint.write_bytes(b"checkpoint")
    wav2vec.write_bytes(b"wav2vec")
    monkeypatch.delenv("SURE_SE_NOMAD_REFERENCE_DIR", raising=False)
    monkeypatch.delenv("SURE_SE_NOMAD_NMR_DIR", raising=False)
    provider = NOMADProvider(checkpoint=str(checkpoint), wav2vec_checkpoint=str(wav2vec))
    with pytest.raises(RuntimeError, match="SURE_SE_NOMAD_REFERENCE_DIR"):
        provider._validate_assets()


def test_estoi_and_lsd_match_local_score_definitions(tmp_path: Path) -> None:
    pystoi = pytest.importorskip("pystoi")
    librosa = pytest.importorskip("librosa")
    time = np.arange(32000, dtype=np.float32) / 16000
    clean = 0.3 * np.sin(2 * np.pi * 440 * time) + 0.1 * np.sin(2 * np.pi * 530 * time)
    enhanced = clean + 0.05 * np.sin(2 * np.pi * 1700 * time)
    clean_path, enhanced_path = tmp_path / "clean.wav", tmp_path / "enhanced.wav"
    _wav(clean_path, clean)
    _wav(enhanced_path, enhanced)
    from sure_eval.evaluation.nodes.scoring._full_reference_audio import read_audio_mono

    reference, _ = read_audio_mono(str(clean_path))
    estimate, _ = read_audio_mono(str(enhanced_path))
    expected_estoi = float(pystoi.stoi(reference, estimate, 16000, extended=True))
    assert ESTOIProvider()(str(enhanced_path), str(clean_path))["estoi"] == pytest.approx(expected_estoi)
    est_db = 20 * np.log10(np.maximum(np.abs(librosa.stft(estimate, n_fft=512, hop_length=128, win_length=512)), 1e-7))
    ref_db = 20 * np.log10(np.maximum(np.abs(librosa.stft(reference, n_fft=512, hop_length=128, win_length=512)), 1e-7))
    expected_lsd = float(np.sqrt(np.mean((est_db - ref_db) ** 2)))
    assert LSDProvider()(str(enhanced_path), str(clean_path))["lsd"] == pytest.approx(expected_lsd)
    assert LSDProvider()(str(clean_path), str(clean_path))["lsd"] == pytest.approx(0.0)


def test_nsrmr_matches_normalized_srmrpy_mode(tmp_path: Path) -> None:
    srmrpy = pytest.importorskip("srmrpy")
    from sure_eval.evaluation.nodes.scoring.nsrmr.node import NSRMRProvider
    from sure_eval.evaluation.nodes.scoring._full_reference_audio import read_audio_mono

    time = np.arange(64000, dtype=np.float32) / 16000
    path = tmp_path / "speech.wav"
    _wav(path, 0.25 * np.sin(2 * np.pi * (140 * time + 60 * time * time)))
    waveform, _ = read_audio_mono(str(path))
    expected = float(srmrpy.srmr(waveform, 16000, fast=True, norm=True)[0])
    result = NSRMRProvider()(str(path))
    assert result["score"] == pytest.approx(expected)
    assert result["unit"] == "ratio"


def test_lsd_resamples_non_16k_audio_like_local_leaderboard(tmp_path: Path) -> None:
    librosa = pytest.importorskip("librosa")
    from sure_eval.evaluation.nodes.scoring._full_reference_audio import read_audio_mono

    time = np.arange(48000, dtype=np.float32) / 48000
    clean_path, enhanced_path = tmp_path / "clean48.wav", tmp_path / "enhanced48.wav"
    _wav(clean_path, 0.25 * np.sin(2 * np.pi * 400 * time), 48000)
    _wav(enhanced_path, 0.25 * np.sin(2 * np.pi * 400 * time) + 0.03 * np.sin(2 * np.pi * 1800 * time), 48000)
    clean, rate = read_audio_mono(str(clean_path))
    enhanced, _ = read_audio_mono(str(enhanced_path))
    clean = librosa.resample(clean, orig_sr=rate, target_sr=16000)
    enhanced = librosa.resample(enhanced, orig_sr=rate, target_sr=16000)
    spec_clean = np.abs(librosa.stft(clean, n_fft=512, hop_length=128, win_length=512))
    spec_enhanced = np.abs(librosa.stft(enhanced, n_fft=512, hop_length=128, win_length=512))
    expected = float(np.sqrt(np.mean((20 * np.log10(np.maximum(spec_clean, 1e-7)) - 20 * np.log10(np.maximum(spec_enhanced, 1e-7))) ** 2)))
    assert LSDProvider()(str(enhanced_path), str(clean_path))["lsd"] == pytest.approx(expected)


def test_component_selection_rejects_missing_and_nonfinite_fields() -> None:
    rows = [("a", "a.wav"), ("b", "b.wav")]

    class Provider:
        def score_batch(self, values, *, metric_name):
            assert values == rows
            return [{"P808_MOS": 3.0}, {"P808_MOS": 4.0}]

    from sure_eval.evaluation.nodes.scoring.dnsmos_p808 import score_dnsmos_p808

    result = score_dnsmos_p808(rows, provider=Provider())
    assert result.details["result"]["score"] == 3.5
    assert result.details["result"]["num_samples"] == 2
    with pytest.raises(KeyError, match="SIG"):
        ScoreComponentProvider(lambda *_: {"BAK": 3}, field="SIG")("a.wav")
    with pytest.raises(ValueError, match="non-finite"):
        ScoreComponentProvider(lambda *_: {"SIG": math.nan}, field="SIG")("a.wav")


def test_lps_denominator_and_empty_reference(monkeypatch) -> None:
    assert levenshtein("abc", "adc") == 1
    assert levenshtein("", "abc") == 3
    provider = LPSProvider(model_dir="unused")
    monkeypatch.setattr(provider, "_wave", lambda path: np.array([0.1], dtype=np.float32))
    phonemes = iter(("ab", "abcd"))
    monkeypatch.setattr(provider, "_phonemes", lambda wave: next(phonemes))
    assert provider("enhanced.wav", "clean.wav")["lps"] == 0.5
    phonemes = iter(("ab", ""))
    monkeypatch.setattr(provider, "_phonemes", lambda wave: next(phonemes))
    with pytest.raises(ValueError, match="empty reference"):
        provider("enhanced.wav", "clean.wav")


def test_new_routes_require_reference_only_when_needed(tmp_path: Path) -> None:
    from sure_eval.evaluation.audio_samples import SampleJsonlError, load_se_samples_jsonl

    enhanced = tmp_path / "enhanced.wav"
    enhanced.write_bytes(b"test")
    samples = tmp_path / "samples.jsonl"
    samples.write_text(json.dumps({"sample_id": "a", "enhanced_audio": "enhanced.wav"}) + "\n")
    for metric in ("estoi", "lsd", "lps", "visqol"):
        with pytest.raises(SampleJsonlError, match="reference_audio is required"):
            load_se_samples_jsonl(samples, metrics=(metric,))
    assert len(load_se_samples_jsonl(samples, metrics=("nisqa_mos", "squim_stoi", "nsrmr"))) == 1
    assert len(load_se_samples_jsonl(samples, metrics=("sq_ast_mos",))) == 1


def test_real_runtime_selects_shared_model_providers_without_loading_assets() -> None:
    from sure_eval.evaluation.audio_runtime import build_se_runtime

    runtime = build_se_runtime(
        metrics=("estoi", "lsd", "lps", "visqol", "nomad", "dnsmos_p808", "dnsmos_sig",
                 "squim_pesq", "squim_stoi", "squim_mos", "scoreq",
                 "nisqa_mos", "nisqa_noi", "nsrmr", "sq_ast_mos", "sq_ast_noi"),
        device="cpu",
    )
    assert set(runtime["reference_providers"]) == {"estoi", "lsd", "lps", "visqol"}
    mos = runtime["mos_providers"]
    assert set(mos) >= {"dnsmos_p808", "dnsmos_sig", "squim_pesq", "squim_stoi",
                        "squim_mos", "scoreq", "nomad", "nisqa_mos", "nisqa_noi", "nsrmr",
                        "sq_ast_mos", "sq_ast_noi"}
    assert mos["dnsmos_p808"] is mos["dnsmos_sig"]
    assert mos["squim_pesq"] is mos["squim_stoi"]
    assert mos["nisqa_mos"] is mos["nisqa_noi"]
    assert mos["sq_ast_mos"] is mos["sq_ast_noi"]


def test_sq_ast_five_dimensions_have_distinct_routes_and_per_sample_scores() -> None:
    from sure_eval.evaluation.tasks.se.pipeline import evaluate_se_samples
    from sure_eval.evaluation.tasks.se.types import SESample

    fields = ("sq_ast_mos", "sq_ast_noi", "sq_ast_dis", "sq_ast_col", "sq_ast_loud")
    values = {field: 3.0 + index * 0.2 for index, field in enumerate(fields)}
    def provider(prediction, reference="", **_):
        return dict(values)
    report = evaluate_se_samples(
        [SESample(enhanced_audio="enhanced.wav", sample_id="example")],
        metrics=fields,
        mos_providers={field: provider for field in fields},
    )

    assert report.pipeline_id == "se.any.multi." + "__".join(
        f"{field}.{field}_v1" for field in fields
    )
    assert report.computation_node_ids == tuple(f"scoring/{field}" for field in fields)
    for field in fields:
        assert report.details["results"][field]["score"] == pytest.approx(values[field])
        assert report.details["results"][field]["per_sample"][0][field] == pytest.approx(values[field])
        assert report.details["results"][field]["higher_is_better"] is True
    assert report.details["results"]["sq_ast_mos"]["unit"] == "predicted_mos"
    for field in fields[1:]:
        assert report.details["results"][field]["unit"] == "predicted_acr_score"


def test_sq_ast_selected_head_scaling_and_cache(monkeypatch, tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from sure_eval.evaluation.nodes.scoring._sq_ast import SQASTProvider

    provider = SQASTProvider(
        dimensions=("noi",), weights_dir=str(tmp_path),
        base_model_dir=str(tmp_path), device="cpu",
    )
    calls = []

    class Extractor:
        def __call__(self, waveform, *, sampling_rate, return_attention_mask, return_tensors):
            assert sampling_rate == 48000
            assert return_attention_mask is True
            assert return_tensors == "pt"
            calls.append("features")
            return {"input_values": torch.ones((1, 1024, 128))}

    class Model:
        def __call__(self, features):
            assert features.shape == (1, 1024, 128)
            calls.append("model")
            return torch.tensor([0.5])

    monkeypatch.setattr(provider, "_load_models", lambda: (Extractor(), {"noi": Model()}))
    monkeypatch.setattr(provider, "_load_waveform", lambda path: torch.ones(48000))
    first = provider("sample.wav")
    second = provider("sample.wav")

    assert first["sq_ast_noi"] == pytest.approx(3.0)
    assert second == first
    assert calls == ["features", "model"]


def test_sq_ast_default_head_path_matches_atomic_node(monkeypatch) -> None:
    from sure_eval.evaluation.nodes.scoring._sq_ast import SQASTProvider

    monkeypatch.delenv("SURE_SE_SQ_AST_WEIGHTS_DIR", raising=False)
    monkeypatch.delenv("SURE_SE_SQ_AST_NOI_CKPT", raising=False)
    provider = SQASTProvider(dimensions=("noi",), device="cpu")
    assert provider._weight_path("noi").parts[-4:] == (
        "sq_ast_noi", "checkpoints", "weights", "noi.pth"
    )
    assert provider.base_model_dir.parts[-3:] == (
        "sq_ast_noi", "checkpoints", "ast-base"
    )


def test_sq_ast_registry_metric_uses_selected_dimension() -> None:
    from sure_eval.evaluation.registry import MetricRegistry

    metric = MetricRegistry.get_metric(
        "sq_ast_noi",
        score_provider=lambda prediction, reference="": {"sq_ast_noi": 4.1},
    )
    result = metric.calculate("enhanced.wav", "")
    assert result.metric_name == "sq_ast_noi"
    assert result.score == pytest.approx(4.1)
    assert result.details["per_sample"][0]["sq_ast_noi"] == pytest.approx(4.1)


def test_sq_ast_cli_describe_run_report_preserves_five_slots(monkeypatch, tmp_path: Path) -> None:
    import sure_eval.evaluation.audio_runtime as runtime

    metrics = ("sq_ast_mos", "sq_ast_noi", "sq_ast_dis", "sq_ast_col", "sq_ast_loud")
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"provider fixture")
    samples = tmp_path / "samples.jsonl"
    samples.write_text(json.dumps({"sample_id": "a", "enhanced_audio": "sample.wav"}) + "\n")
    values = {metric: 3.0 + index * 0.1 for index, metric in enumerate(metrics)}

    def providers(*, metrics: tuple[str, ...], device="cuda", cache_dir=None):
        assert metrics == ("sq_ast_mos", "sq_ast_noi", "sq_ast_dis", "sq_ast_col", "sq_ast_loud")
        def provider(prediction, reference="", **_):
            return dict(values)
        return {"reference_providers": {}, "mos_providers": {metric: provider for metric in metrics}}

    monkeypatch.setattr(runtime, "build_se_runtime", providers)
    runner = CliRunner()
    pipeline = tmp_path / "pipeline.json"
    describe = runner.invoke(app, [
        "metric", "describe", "se", "--metrics", ",".join(metrics),
        "--output", str(pipeline), "--json",
    ])
    assert describe.exit_code == 0, describe.stdout
    spec = json.loads(pipeline.read_text())
    assert spec["metrics"] == list(metrics)
    assert [slot["metric"] for slot in spec["pipeline"]] == list(metrics)
    output = tmp_path / "out"
    run = runner.invoke(app, [
        "metric", "run", "--pipeline", str(pipeline),
        "--samples-jsonl", str(samples), "--output-dir", str(output),
        "--device", "cpu", "--json",
    ])
    assert run.exit_code == 0, run.stdout
    report = json.loads((output / "report.json").read_text())
    description = json.loads((output / "pipeline_description.json").read_text())
    assert report["pipeline_id"] == description["pipeline_id"] == spec["pipeline_id"]
    assert list(report["details"]["results"]) == list(metrics)
    for metric in metrics:
        assert report["details"]["results"][metric]["score"] == pytest.approx(values[metric])
        assert len(report["details"]["results"][metric]["per_sample"]) == 1


def test_squim_mos_and_scoreq_are_aggregated_as_distinct_se_scores() -> None:
    from sure_eval.evaluation.tasks.se.pipeline import evaluate_se_samples
    from sure_eval.evaluation.tasks.se.types import SESample

    report = evaluate_se_samples(
        [
            SESample(enhanced_audio="a.wav", reference_audio="clean.wav", sample_id="a"),
            SESample(enhanced_audio="b.wav", reference_audio="clean.wav", sample_id="b"),
        ],
        metrics=("squim_mos", "scoreq"),
        mos_providers={
            "squim_mos": lambda prediction, reference="", **_: {"squim_mos": 3.0 if prediction == "a.wav" else 4.0},
            "scoreq": lambda prediction, reference="", **_: {"scoreq": 2.0 if prediction == "a.wav" else 4.0},
        },
    )

    assert report.pipeline_id == "se.any.multi.squim_mos.squim_mos_v1__scoreq.scoreq_v1"
    assert report.details["results"]["squim_mos"]["score"] == 3.5
    assert report.details["results"]["scoreq"]["score"] == 3.0
    assert report.details["results"]["squim_mos"]["unit"] == "predicted_mos"
    assert report.details["results"]["scoreq"]["unit"] == "predicted_mos"


def test_squim_subjective_provider_uses_non_matching_reference(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    from sure_eval.evaluation.nodes.scoring._squim_subjective import SQUIMSubjectiveProvider

    provider = SQUIMSubjectiveProvider(non_matching_reference="staged.wav", device="cpu")
    calls = []

    class Model:
        def __call__(self, waveform, reference):
            calls.append((waveform.shape, reference.shape))
            return torch.tensor([4.25])

    monkeypatch.setattr(provider, "_load_audio", lambda path: torch.tensor([0.1, 0.2]))
    monkeypatch.setattr(provider, "_load_reference", lambda: torch.tensor([0.3, 0.4, 0.5]))
    monkeypatch.setattr(provider, "_load_model", lambda: Model())
    result = provider("enhanced.wav")

    assert result["squim_mos"] == pytest.approx(4.25)
    assert calls == [((1, 2), (1, 3))]


def test_scoreq_provider_reads_one_scalar_from_onnx_session(monkeypatch) -> None:
    import numpy as np
    from sure_eval.evaluation.nodes.scoring._scoreq import SCOREQProvider

    provider = SCOREQProvider(checkpoint="staged.onnx", device="cpu")

    class Session:
        def get_inputs(self):
            return [type("Input", (), {"name": "waveform"})()]

        def run(self, outputs, inputs):
            assert outputs is None
            assert inputs["waveform"].shape == (1, 320)
            return [np.array([[3.75]], dtype=np.float32)]

    monkeypatch.setattr(provider, "_load_session", lambda: Session())
    monkeypatch.setattr(provider, "_load_audio", lambda path: np.zeros((1, 320), dtype=np.float32))
    result = provider("enhanced.wav")

    assert result["scoreq"] == pytest.approx(3.75)


def test_scoreq_default_checkpoint_matches_declared_node_asset(monkeypatch) -> None:
    from sure_eval.evaluation.nodes.scoring._scoreq import SCOREQProvider

    monkeypatch.delenv("SURE_SE_SCOREQ_CHECKPOINT", raising=False)
    provider = SCOREQProvider(device="cpu")
    assert Path(provider.checkpoint).parts[-3:] == (
        "scoreq", "checkpoints", "adapt_nr_telephone.onnx"
    )


def test_cli_describe_run_report_preserves_identity(monkeypatch, tmp_path: Path) -> None:
    import sure_eval.evaluation.audio_runtime as runtime

    enhanced, reference = tmp_path / "enhanced.wav", tmp_path / "reference.wav"
    enhanced.write_bytes(b"provider fixture")
    reference.write_bytes(b"provider fixture")
    samples = tmp_path / "samples.jsonl"
    samples.write_text(json.dumps({"sample_id": "a", "enhanced_audio": "enhanced.wav", "reference_audio": "reference.wav"}) + "\n")

    def providers(*, metrics, device="cuda", cache_dir=None):
        assert tuple(metrics) == ("estoi", "dnsmos_p808")
        return {
            "reference_providers": {"estoi": lambda *_: {"estoi": 0.83}},
            "mos_providers": {"dnsmos_p808": lambda *_: {"P808_MOS": 3.5}},
        }

    monkeypatch.setattr(runtime, "build_se_runtime", providers)
    runner = CliRunner()
    routes = runner.invoke(app, ["metric", "routes", "se", "--metric", "dnsmos_p808", "--json"])
    assert routes.exit_code == 0, routes.stdout
    assert json.loads(routes.stdout)["routes"][0]["pipeline_id"] == "se.any.dnsmos_p808.dnsmos_p808_v1"
    pipeline = tmp_path / "pipeline.json"
    describe = runner.invoke(app, ["metric", "describe", "se", "--metrics", "estoi,dnsmos_p808", "--output", str(pipeline), "--json"])
    assert describe.exit_code == 0, describe.stdout
    spec = json.loads(pipeline.read_text())
    expected = "se.any.multi.estoi.estoi_v1__dnsmos_p808.dnsmos_p808_v1"
    assert spec["pipeline_id"] == expected
    output = tmp_path / "out"
    run = runner.invoke(app, ["metric", "run", "--pipeline", str(pipeline), "--samples-jsonl", str(samples), "--output-dir", str(output), "--device", "cpu", "--json"])
    assert run.exit_code == 0, run.stdout
    report = json.loads((output / "report.json").read_text())
    description = json.loads((output / "pipeline_description.json").read_text())
    assert report["pipeline_id"] == description["pipeline_id"] == expected
    assert report["details"]["results"]["estoi"]["score"] == 0.83
    assert report["details"]["results"]["dnsmos_p808"]["score"] == 3.5
    assert report["details"]["results"]["estoi"]["higher_is_better"] is True
    assert report["details"]["results"]["dnsmos_p808"]["unit"] == "predicted_mos"
    assert report["computation_node_ids"] == description["computation_node_ids"]


def test_script_stub_reports_new_atomic_identity(tmp_path: Path) -> None:
    output = tmp_path / "stub.json"
    subprocess.run([
        sys.executable, "scripts/run_se_metric_pipeline.py", "--enhanced-audio", "enhanced.wav",
        "--reference-audio", "clean.wav", "--metrics", "lsd,nsrmr,squim_mos,scoreq", "--stub",
        "--output", str(output),
    ], check=True)
    payload = json.loads(output.read_text())
    assert payload["report"]["pipeline_id"] == (
        "se.any.multi.lsd.lsd_v1__nsrmr.nsrmr_v1__squim_mos.squim_mos_v1__scoreq.scoreq_v1"
    )
    assert payload["metrics"]["lsd"]["score"] == 5.2
    assert payload["metrics"]["nsrmr"]["score"] == 4.1
    assert payload["metrics"]["lsd"]["higher_is_better"] is False
    assert payload["metrics"]["nsrmr"]["unit"] == "ratio"
    assert payload["metrics"]["squim_mos"]["score"] == 3.7
    assert payload["metrics"]["scoreq"]["score"] == 3.8

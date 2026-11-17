"""ViSQOL v3.3.3 speech-mode scoring node."""

from __future__ import annotations

import argparse
import json
import sys

from sure_eval.evaluation.core.types import PipelineNodeResult
from sure_eval.evaluation.nodes.scoring._full_reference_audio import (
    FullReferenceAudioProvider,
    FullReferenceAudioRow,
    ViSQOLProvider,
    score_full_reference_audio_backend,
)

NODE_ID = "scoring/visqol"
NODE_VERSION = "v1"


def score_visqol(
    rows: list[FullReferenceAudioRow],
    *,
    provider: FullReferenceAudioProvider | None = None,
) -> PipelineNodeResult:
    return score_full_reference_audio_backend(
        rows,
        metric_name="visqol",
        node_id=NODE_ID,
        provider=provider or ViSQOLProvider(),
        version=NODE_VERSION,
    )


def _read_rows(path: str) -> list[FullReferenceAudioRow]:
    rows: list[FullReferenceAudioRow] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append(
                (
                    str(row.get("key", "")),
                    str(row.get("enhanced_audio", row.get("prediction_audio", ""))),
                    str(row["reference_audio"]),
                )
            )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score enhanced speech with ViSQOL v3.3.3.")
    parser.add_argument("--prediction-audio")
    parser.add_argument("--reference-audio")
    parser.add_argument("--input-jsonl")
    parser.add_argument("--json", action="store_true", dest="json_output")
    args = parser.parse_args(argv)

    has_pair_arg = bool(args.prediction_audio or args.reference_audio)
    has_complete_pair = bool(args.prediction_audio and args.reference_audio)
    if (args.input_jsonl and has_pair_arg) or (not args.input_jsonl and not has_complete_pair):
        parser.error("use either --input-jsonl or --prediction-audio plus --reference-audio")

    provider = ViSQOLProvider()
    if args.input_jsonl:
        trace = score_visqol(_read_rows(args.input_jsonl), provider=provider)
        payload = {
            "node_id": NODE_ID,
            "version": NODE_VERSION,
            "result": trace.details["result"],
            "keys": trace.details["keys"],
        }
    else:
        result = provider(args.prediction_audio, args.reference_audio)
        payload = {
            "node_id": NODE_ID,
            "version": NODE_VERSION,
            "prediction_audio": args.prediction_audio,
            "reference_audio": args.reference_audio,
            "result": result,
        }
    if args.json_output:
        sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    else:
        print(payload["result"].get("score", payload["result"].get("visqol", "")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

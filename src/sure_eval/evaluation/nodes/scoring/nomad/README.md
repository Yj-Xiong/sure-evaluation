# NOMAD scoring

`scoring/nomad` computes the official NOMAD non-matching-reference distance
between enhanced speech and a fixed pool of clean reference utterances. The
result is the mean Euclidean distance between normalized 256-dimensional
NOMAD embeddings; lower values indicate greater perceptual similarity.

## Route

- Node: `scoring/nomad`
- Version: `v1`
- Pipeline: `se.any.nomad.nomad_v1`
- Input: `enhanced_audio`
- Reference protocol: staged clean non-matching pool

## Setup

Install the optional node environment and stage the two official checkpoint
files. The pool is benchmark data and must be supplied separately:

```bash
export SURE_SE_NOMAD_CHECKPOINT=/path/to/nomad_best_model.pt
export SURE_SE_NOMAD_WAV2VEC_CHECKPOINT=/path/to/wav2vec_small.pt
export SURE_SE_NOMAD_REFERENCE_DIR=/path/to/clean_reference_pool
sure-eval env check --node scoring/nomad
```

`nomad_audio==0.0.8` loads both weights from a private temporary `pt-models`
directory. No checkpoints or clean reference audio are committed to this
repository. The same reference pool must be used for every system in one
benchmark report.

## References

- Ragano, Skoglund, and Hines, “NOMAD: Unsupervised Learning of Perceptual
  Embeddings For Speech Enhancement and Non-Matching Reference Audio Quality
  Assessment,” ICASSP 2024, DOI 10.1109/ICASSP48485.2024.10448028.
- Official implementation: https://github.com/alessandroragano/nomad

NOMAD is MIT licensed. The model checkpoint and reference-pool terms must be
verified before redistribution.

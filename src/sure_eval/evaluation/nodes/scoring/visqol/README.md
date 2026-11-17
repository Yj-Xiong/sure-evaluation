# ViSQOL v3.3.3 scoring

`scoring/visqol` computes the full-reference speech mode of ViSQOL. It compares
each enhanced signal with its matching clean reference and returns the native
MOS-LQO score on a 1–5 scale.

## Route

- Node: `scoring/visqol`
- Version: `v1`
- Pipeline: `se.any.visqol.visqol_v1`
- Input: `enhanced_audio` and `reference_audio`
- Sample rate: 16 kHz, downmixed to mono before scoring

This node calls the C++ CLI in Google's official ViSQOL source at
tag `v3.3.3`, commit `c3aa2e498e0f7f14202643594335a0b9ee40bdd9`.
It explicitly enables the deep-lattice speech mapper. The separate
`visqol-python` package is not used.

## Setup

Build the official CLI in a cluster job, following Google's build
instructions. A checkout at the pinned commit should provide the model file.
The following illustrates the required source layout; do not build on a debug
machine:

```bash
git clone https://github.com/google/visqol.git /path/to/google-visqol
git -C /path/to/google-visqol checkout c3aa2e498e0f7f14202643594335a0b9ee40bdd9
cd /path/to/google-visqol
bazel build -c opt :visqol
export SURE_SE_VISQOL_BINARY=/path/to/google-visqol/bazel-bin/visqol
export SURE_SE_VISQOL_LATTICE_MODEL=/path/to/google-visqol/model/lattice_tcditugenmeetpackhref_ls2_nl60_lr12_bs2048_learn.005_ep2400_train1_7_raw.tflite
sure-eval env check --node scoring/visqol
```

`sure-eval env setup --node scoring/visqol` installs only the Python audio
decoding and resampling dependencies. It does not build or substitute the
official binary. The route fails if the binary or model is unavailable.

The provider converts each pair to aligned, mono 16 kHz PCM16 WAV and calls
the official CLI with `--use_speech_mode=true`, `--use_lattice_model=true`,
and an explicit model path. It reads the single-row `--results_csv` output,
validates the requested pair and MOS-LQO, and records SHA-256 hashes of the
binary and model alongside the score.

## References

- Google ViSQOL release: https://github.com/google/visqol/releases/tag/v3.3.3
- Official CLI flags and format: https://github.com/google/visqol/blob/v3.3.3/README.md
- Chinen et al., “ViSQOL v3: An Open Source Production Ready Objective Speech
  and Audio Metric,” QoMEX 2020

ViSQOL is a full-reference metric. It must not be used as a no-reference score
when the matching clean recording is unavailable.

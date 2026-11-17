# SE Task

The SE task evaluates speech enhancement outputs.

Input rows use `enhanced_audio` for the system output, `noisy_audio` for the
model input when available, and `reference_audio` for clean speech when
full-reference metrics are selected.

Supported metrics:

| Canonical metric | Execution selector | Node |
| --- | --- | --- |
| `si_sdr` | `si_sdr` / `si-sdr` | `scoring/si_sdr` |
| `stoi` | `stoi` | `scoring/stoi` |
| `estoi` | `estoi` | `scoring/estoi` |
| `lsd` | `lsd` | `scoring/lsd` |
| `lps` | `lps` | `scoring/lps` |
| `pesq` | `pesq` | `scoring/pesq` |
| `visqol` | `visqol` | `scoring/visqol` |
| `nomad` | `nomad` | `scoring/nomad` |
| `dnsmos` | `dnsmos` | `scoring/dnsmos` |
| `dnsmos_p808` | `dnsmos_p808` | `scoring/dnsmos_p808` |
| `dnsmos_sig` | `dnsmos_sig` | `scoring/dnsmos_sig` |
| `dnsmos_bak` | `dnsmos_bak` | `scoring/dnsmos_bak` |
| `wv_mos` | `wv_mos` / `wv-mos` | `scoring/wv_mos` |
| `utmos` | `utmos` | `scoring/utmos` |
| `nsrmr` | `nsrmr` | `scoring/nsrmr` |
| `squim_stoi` / `squim_pesq` / `squim_si_sdr` | same as canonical | corresponding `scoring/squim_*` node |
| `nisqa_mos` / `nisqa_noi` / `nisqa_dis` / `nisqa_col` / `nisqa_loud` | same as canonical | corresponding `scoring/nisqa_*` node |

`lsd` and `nomad` are lower-is-better; other metrics are higher-is-better. `visqol` is a
full-reference speech-mode MOS-LQO on the native 1–5 scale and uses Google's
v3.3.3 C++ CLI with its speech lattice mapper.
`nomad` is a lower-is-better non-matching-reference embedding distance. It
requires a fixed clean reference pool staged through
`SURE_SE_NOMAD_REFERENCE_DIR`; the pool is not the paired `reference_audio`
role from full-reference routes.
All aggregate by arithmetic mean over samples. The default bundle is unchanged.
New metrics are opt-in. `estoi` and `lsd` require clean `reference_audio`;
`lps` also requires that role. See [the task guide](../../../../../docs/tasks/se.md)
for exact pipeline IDs, score definitions, runtime assets, and licenses.

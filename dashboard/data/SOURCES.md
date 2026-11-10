# Where each file in data/ comes from

Copied on 2026-11-10 by `tools/import_paper.py` (atoms byte for byte; the rest extracted, as described below).
`$RSI` = `~/Projects/ambient/research/rsi-inference-design`; `$PAPER` = `~/Projects/rsi/paper`.

| File | Origin | How |
|---|---|---|
| `atoms/e1-race-check-dots-ordered.json` | `$RSI/paper2/atoms/e1-race-check-dots-ordered.json` | copy (sha256 c7b73554ff0e, source c7b73554ff0e) |
| `atoms/f4-fidelity-systems.json` | `$RSI/paper2/atoms/f4-fidelity-systems.json` | copy (sha256 6213e5e3984a, source 6213e5e3984a) |
| `atoms/m-check-phases-3sys.json` | `$RSI/paper2/atoms/m-check-phases-3sys.json` | copy (sha256 ed01550ec2a1, source ed01550ec2a1) |
| `atoms/m4-model-speed-replay.json` | `$RSI/paper2/atoms/m4-model-speed-replay.json` | copy (sha256 28a175459476, source 28a175459476) |
| `atoms/r-decode-d16x128k.json` | `$RSI/paper2/atoms/r-decode-d16x128k.json` | copy (sha256 fc8611ae8025, source fc8611ae8025) |
| `atoms/r-decode-d1x32k.json` | `$RSI/paper2/atoms/r-decode-d1x32k.json` | copy (sha256 01e96e85b035, source 01e96e85b035) |
| `atoms/r-decode-d32x32k.json` | `$RSI/paper2/atoms/r-decode-d32x32k.json` | copy (sha256 a4fe85d760bb, source a4fe85d760bb) |
| `atoms/r-decode-d8x32k.json` | `$RSI/paper2/atoms/r-decode-d8x32k.json` | copy (sha256 c78efb613df4, source c78efb613df4) |
| `atoms/r-gqa-kernel.json` | `$RSI/paper2/atoms/r-gqa-kernel.json` | copy (sha256 5f2264119727, source 5f2264119727) |
| `atoms/r-megatron.json` | `$RSI/paper2/atoms/r-megatron.json` | copy (sha256 95f5c5deb835, source 95f5c5deb835) |
| `atoms/r-mla-kernel.json` | `$RSI/paper2/atoms/r-mla-kernel.json` | copy (sha256 c2bfa9e3e062, source c2bfa9e3e062) |
| `atoms/r-moe-bwd-kernel.json` | `$RSI/paper2/atoms/r-moe-bwd-kernel.json` | copy (sha256 5561cfff482a, source 5561cfff482a) |
| `atoms/r-moe-fwd-kernel.json` | `$RSI/paper2/atoms/r-moe-fwd-kernel.json` | copy (sha256 f8f838216ca1, source f8f838216ca1) |
| `atoms/r-nanogpt.json` | `$RSI/paper2/atoms/r-nanogpt.json` | copy (sha256 a17f568cdbbf, source a17f568cdbbf) |
| `atoms/r-omni.json` | `$RSI/paper2/atoms/r-omni.json` | copy (sha256 62645ba91b81, source 62645ba91b81) |
| `atoms/r-prefill-p16k.json` | `$RSI/paper2/atoms/r-prefill-p16k.json` | copy (sha256 c3133c93177c, source c3133c93177c) |
| `atoms/r-prefill-p32k.json` | `$RSI/paper2/atoms/r-prefill-p32k.json` | copy (sha256 a114384fa408, source a114384fa408) |
| `atoms/r-prefill-p64k.json` | `$RSI/paper2/atoms/r-prefill-p64k.json` | copy (sha256 d1da24e0db06, source d1da24e0db06) |
| `atoms/r-prefill-p8k.json` | `$RSI/paper2/atoms/r-prefill-p8k.json` | copy (sha256 443cbfd1df43, source 443cbfd1df43) |
| `atoms/r-t2av-seeds.json` | `$RSI/paper2/atoms/r-t2av-seeds.json` | copy (sha256 a6b93c8b4425, source a6b93c8b4425) |
| `atoms-index.json` | `$RSI/paper2/atoms/index.json` | the entries of the copied atoms |
| `picks.json` | `$RSI/paper2/publish/runs-index.json` (main-vllm, main-systems = paper Fig. 5, 6), `$RSI/paper2/publish/index.json` (p4-low-overhead, p5-kernels-appendix = Oct 29 draft Fig. 12, 13), `$RSI/paper2/paper2.db` table `picks` rows `runs:*` | the paper's own run selection, unchanged |
| `fig1-ladder.json` | `$PAPER/data/nemotron-fig1-serve.9c1a829.json` | rows computed as `$PAPER/figures/fig1_stacked.py` does (Fig. 1, and the vLLM panel of Fig. 3) |
| `fig8-checks.json` | `$RSI/paper2/publish/index.json` entry `p7-checks` | copy of its rows (paper Fig. 8) |
| `table2-ablation.json` | `$RSI/paper-figs-20261024/data/ablations.json` | `by_group["DeepSeek v4-v5 (s6-s11)"]` (paper Table 2) |
| `decode-checks.json` | `$RSI/paper-figs-20261024/raw/decode/ledger-d{1,2}.jsonl`, `score-dec{1,2}.out` | every error-free 4-case bench, as speedup over the official start (the decode atoms carry no check dots) |
| `steps.json` | race ledgers and gate files under `$RSI/paper-figs-20261024/raw/`, `$RSI/megatron-lab-20261028/data/race5/`, kernel and speedrun task files in `$RSI/paper-figs-20261024/data/`; subjects from `raw/commit-subjects.txt` and `data/deepseek_ideas.json` | each best-so-far step matched to the ledger row with the exact same time; unmatched steps are left out |
| `paper_values.json` | `$HOME/Downloads/fastrsi (11).pdf`, `fastrsi (3).pdf` | transcribed by reading the PDFs |

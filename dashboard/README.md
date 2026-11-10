# FastRSI results dashboard

One page that shows the paper (LiveLab-MLSys draft)'s three claims with the paper's own figure data, made interactive:

| Band | Claim | Paper figures |
|---|---|---|
| A. Reduce cold start | Keeping state warm removes most of a restart: 228 s to 23.6 s on vLLM | Fig. 1a, Fig. 3 |
| B. Reduce evaluation time | A live check returns 1.8-5.8x sooner and stays faithful | Fig. 1b, Fig. 7, Fig. 8, Fig. 9 |
| C. End-to-end speedup | FastRSI reaches the restart baseline's final score sooner | Fig. 5, Fig. 6, Fig. 10, Table 2; kernels and nanoGPT from the Oct 29 draft (Fig. 12, 13) |

The live arm is labelled FastRSI; the atoms' run names ("LiveLab #5") are shown as "FastRSI #5".
No new GPU runs. Every number comes from `data/`, which holds copies of the paper's figure data and a
`SOURCES.md` that names where each file came from.

## Run it

```sh
python3 dashboard/build.py            # data/ -> dashboard.json, prints the paper check
python3 dashboard/serve.py            # http://127.0.0.1:8742
```

`serve.py` is read-only and binds to `127.0.0.1:8742` only. Pass `--port N` to change the port.

Other outputs of `build.py`:

```sh
python3 dashboard/build.py --standalone out/dashboard.html   # one file: HTML, CSS, JS and data inlined
python3 dashboard/build.py --table evidence/dashboard/verification.md
```

The standalone file needs no server; it loads its two web fonts from Google Fonts and falls back to
system fonts offline.

## What you can do on the page

- Pick a workload: vLLM prefill (8K-64K), vLLM decode (four batch shapes), Megatron-LM, FastVideo-T2VA,
  FastVideo-Ref2VA, four GPU kernels (GQA, MLA, MoE forward, MoE backward) and the nanoGPT speedrun.
- Hover a step of a best-so-far line to see when it landed, the score, the commit and its subject line.
- Turn each arm on or off, show every check the run measured as dots behind the line, and show the
  side's other picked runs as thin lines.
- Choose which run each side draws (default: the run the paper draws).
- Drag across the race chart to zoom in on a time window; double-click or press "Reset zoom" to undo.
- Read the "N x sooner" dimension line: where FastRSI first reaches the restart baseline's final score.
- Click a band's heading to open or close its phase breakdown.

## Which runs are drawn

The paper's run selection is used as-is (`data/picks.json`):

- vLLM, Megatron-LM, FastVideo: the runs picked on the paper's `/runs` page (hidden runs are never shown),
  and the best run per side as named in `publish/runs-index.json` (highest mean speedup over the race).
  These are the runs in Fig. 5 and Fig. 6.
- Kernels: every race, as in the Oct 29 draft's Fig. 13; by default race 1 of each side, except the MoE
  backward kernel, which defaults to the Fig. 12 pair (Codex race 1, FastRSI race 5).
- nanoGPT: race 1 on dedicated GPUs (Fig. 12); race 2 on shared GPUs is available as another run.

## Paper check

`build.py` compares every chart against the value printed in the paper (`data/paper_values.json`,
transcribed from `fastrsi (11).pdf` and, for kernels and nanoGPT, `fastrsi (3).pdf`) and prints one row
per number. A mismatch is reported and drawn on the page as a note; the data is never changed to agree.

## Updating the data

`tools/import_paper.py` is the only script that reads outside this folder. Run it on the machine that
holds the paper sources, then rebuild:

```sh
python3 dashboard/tools/import_paper.py --rsi ~/Projects/ambient/research/rsi-inference-design --paper ~/Projects/rsi/paper
python3 dashboard/build.py
```

## Files

| File | Role |
|---|---|
| `build.py` | stdlib only: `data/` -> `dashboard.json`, paper check, standalone export |
| `serve.py` | stdlib `http.server` on 127.0.0.1:8742, serves this folder |
| `index.html`, `style.css`, `app.js` | the page; vanilla JS, inline SVG, no build step |
| `data/` | copied figure data, `paper_values.json`, `SOURCES.md` |
| `tools/import_paper.py` | refreshes `data/` from the paper sources |

## Not shown

- Fig. 2 (wall-clock split of a 6-hour vLLM run at faster decode speeds): no data file for the vLLM
  version was found; only the older FastVideo Omni version exists (`paper/data/omni-wallclock.json`).
- Table 1 (full-evaluation time per task): prose only.
- Commit subjects are missing where no git log was saved: Megatron, kernels, nanoGPT, decode race 1 and
  FastVideo show the commit hash only, or nothing when the step could not be matched to a ledger row.

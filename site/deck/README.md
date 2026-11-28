# rsibox.fastrsi.ai

The interactive RSIBox website: one static page (`index.html`, inline CSS and JS, no build step).
Each section is a small live demo of one mechanism: hot-reload, tier routing, the edit lifecycle,
undo, crash recovery, the agent pool, and the results.

```bash
python3 -m http.server 8761 --bind 127.0.0.1 --directory site/deck
```

Then open http://127.0.0.1:8761/.

## Figures

`fig.js` and `fig.css` turn each `<div data-fig="...">` into an interactive version of a figure
from the RSIBox blog post, drawn in the blog's style (white plate, rust start-up, blue RSIBox):

| `data-fig` | Blog figure | Reads |
|---|---|---|
| `ladder` | Fig. 2, restart overhead ladder | `eval_ladder.csv`, `eval_phases.csv`, `ladder_rungs.csv`, `race_workloads.csv` |
| `bottleneck` | Fig. 1, evaluation dominates | `fig2_breakdown.csv` |
| `engines` | Fig. 4, six systems on a log scale | values in `fig.js` (from the post) |
| `raceGrid` | Fig. 5, end-to-end races: all six races, every run | `race_runs.csv`; tile names, edit counts and the Miles pick in `fig.js` (`RACE_TILES`, from the post) |

`data/` is a link to `site/landing/dash/data/`, so the site and the landing figures share one
copy of every number. The Pages workflow copies the real files in (`cp -RL`).
`data-extra/` holds race rows for the deck only, read on top of `data/` by `raceGrid` (same file
names and headers; `data/` wins for a workload in both): the FastVideo (T2VA) race, restored from
`git show cabc09b8^:site/landing/dash/data/` because the blog still reports it, and the SGLang and
Miles runs the blog's Figure 5 draws (SGLang r2 and r3; Miles r1 and r2 on RSIBox, r3 and r4 on
Baseline), taken from the race records the blog figure was drawn from.

## Deploy

`.github/workflows/pages.yml` publishes this folder to GitHub Pages on every push to `main`
that touches `site/deck/`. The custom domain is `rsibox.fastrsi.ai`:

1. Repo settings, Pages: source "GitHub Actions", custom domain `rsibox.fastrsi.ai`, enforce HTTPS.
2. DNS (Cloudflare, zone `fastrsi.ai`): `CNAME rsibox -> hao-ai-lab.github.io`, proxy off (DNS only)
   until GitHub has issued the certificate.

## Rules

- A number on the page comes from the repo README or `docs/`. Code edits shown in the demos are
  illustrative and labelled so on the page.
- Fonts load from Google Fonts; everything else is inline.

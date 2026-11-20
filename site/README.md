# FastRSI site

The public site: a home page that argues the case top to bottom.
Static files, no dependencies, no build tools beyond Python 3.

```bash
python3 site/build.py
python3 -m http.server 8760 --bind 127.0.0.1 --directory site/public
```

Then open http://127.0.0.1:8760/ (home).

| Folder | What it holds |
|---|---|
| `landing/src/` | The home page: `page.html` holds all words and layout; `story.js` the scroll story; `site.js` the landing figures, popups and the loop; `story.css`, `tokens.css` the look. |
| `landing/dash/` | The three results figures (`files/`) and their data (`data/*.csv`). The figures read every number from these CSVs. |
| `facts/` | Sourced facts the page reads at build time: per-engine mechanisms, replayed upstream PRs, the static coverage survey, fault cases. |
| `build.py` | Builds the page into `public/` (git-ignored) as a self-contained HTML file. |

## Rules

- A number on a page comes from a file in `landing/dash/data/` or `facts/`. A value that is illustrative is labelled so on the page. A fact that is not recorded is shown as "not measured yet".
- The built page is ASCII only, so it reads the same whatever charset a host decodes it with.
- The page loads d3 from cdnjs and fonts from Google Fonts; everything else is inline.
- To change wording, edit `landing/src/page.html` and rebuild.

## Case studies

The home page's case studies read three files in `facts/`:

| File | What it holds |
|---|---|
| `case-timing.json` | Measured edit-to-check-result time per case, for a plain restart and for FastRSI, with the phases that add up to each total. Written by `python3 site/facts/make-case-timing.py <cases dir> ...` from the measurement tables. |
| `case-copy.json` | The four text blocks per case, keyed by pull request number or agent-edit id. |
| `pr-diffs/` | The upstream diff of each replayed pull request (non-test files). |

A case in `case-timing.json` that matches no pull request in `prs.json` is shown as one of our own agent edits, with the diff taken from the timing file.

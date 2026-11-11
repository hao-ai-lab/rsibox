# FastRSI site

The public site: a home page that argues the case top to bottom, and an audit page with the detail.
Static files, no dependencies, no build tools beyond Python 3.

```bash
python3 site/build.py
python3 -m http.server 8760 --bind 127.0.0.1 --directory site/public
```

Then open http://127.0.0.1:8760/ (home) and http://127.0.0.1:8760/audit/ (audit).

| Folder | What it holds |
|---|---|
| `landing/src/` | The home page: `page.html` holds all words and layout; `story.js` the scroll story; `site.js` the landing figures, popups and the loop; `story.css`, `tokens.css` the look. |
| `landing/dash/` | The three results figures (`files/`) and their data (`data/*.csv`). The figures read every number from these CSVs. |
| `audit/src/` | The audit page. |
| `facts/` | Sourced facts both pages read at build time: per-engine mechanisms, replayed upstream PRs, the static coverage survey, fault cases. |
| `build.py` | Builds both pages into `public/` (git-ignored) as two self-contained HTML files. |

## Header tabs

The header shows Overview and GitHub. The audit views (Supported PRs, Coverage, Conventions, Faults) can also appear there as tabs; they are off until those pages are ready. `python3 site/build.py --tabs` turns them on.

## Rules

- A number on a page comes from a file in `landing/dash/data/` or `facts/`. A value that is illustrative is labelled so on the page. A fact that is not recorded is shown as "not measured yet".
- The built pages are ASCII only, so they read the same whatever charset a host decodes them with.
- `audit/build.py` fails the build if a banned word or a private path appears in the audit page.
- The pages load d3 from cdnjs and fonts from Google Fonts; everything else is inline.
- To change wording, edit `landing/src/page.html` or `audit/src/page.html` and rebuild.

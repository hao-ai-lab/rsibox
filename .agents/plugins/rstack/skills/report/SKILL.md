---
name: report
description: Inline visual change-report standard — print a 1-line file description and concrete verbatim Before -> After excerpts directly on screen so a human can review without opening files.
---

# Inline Visual Change Report (`/report`)

Never hide substantive code, doc, or config changes behind bare file paths or vague summaries. Present a self-contained inline report directly in your response:

1. **One-Line File Summary**: For each key modified or added file, print a single clear line stating the file link and what changed and why.
2. **Verbatim `Before -> After` Excerpts**: Immediately below each file summary, show concrete verbatim `Before` and `After` snippets (or exact added excerpt for new files) directly on screen so a human reviewer can judge the diff without opening the file.
3. **Exact Verification Ledger**: List the exact commands run after the final edit, exact pass/skip counts, and any `Not verified` gaps.
4. **No Paraphrased Diffs**: Quote real lines from the workspace; never fabricate or approximate excerpts.

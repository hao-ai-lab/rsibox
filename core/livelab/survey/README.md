# Survey: how much of an engine can be edited live

A static scan of an engine's upstream source tree. For every function it says which function
group it is in, how an edit to it would be installed, and whether the engine's lab has that
install path. CPU only, standard library only; it parses text with `ast` and imports nothing
of the engine.

It answers "how much of the tree has a live install path in this lab". It does not answer
"how much of what my workload runs": see [What it cannot tell](#what-it-cannot-tell).

## Contract

- **Input:** an upstream tree, an engine's rules file, and the lab's files (the working tree,
  or one commit).
- **Unit:** a function a module or a class body defines. A nested function or lambda counts
  with the function that encloses it. Code outside every function (imports, constants, class
  bodies) is one `module_level` record per file, in lines only. A native source file
  (`.c .cc .cpp .cxx .cu .cuh .h .hpp`) is one record, counted by file and line, apart from
  the Python shares.
- **Per record:** `group`, `install` (the install class), `capability` (the lab path that
  would install it), `covered`, `rule`, `basis` and `firm`.
- **Output:** one JSON report per engine (`schema: fastrsi-survey/1`): totals and
  per-top-level-module counts, in functions and in lines, per install class, with the
  upstream commit scanned, the scan date, the rule set's version and the rules file's digest.
  `--functions` adds one JSON line per record.
- **Owns:** the framework rules below and the report format. Not the engine's declarations:
  those are the engine's rules file.

### Install classes

From `core/agents/livelab_overview.md` and `INTERFACE.md` ("How install refreshes").

| Group | Install class | Meaning |
|---|---|---|
| OP | `swap` | swap the function body in place |
| GRAPH | `swap_recapture` | swap, then recapture or recompile the graphs that hold it |
| STATE, RESOURCE | `rebuild` | rebuild in place the objects built from it |
| RANKS | `respawn` | a new worker group, forked from the zygote |
| NATIVE (Triton) | `swap_recapture` | rebuild the compiled kernel from its new text (capability `jit_rebuild`) |
| NATIVE (other DSL) | `respawn` | a new process compiles it again |
| NATIVE (C++/CUDA) | `restart` | needs a build; no lab loads a versioned library yet |
| restart | `restart` | startup logic, launch flags, environment variables |

`swap`, `swap_recapture` and `rebuild` are **live** (same process). `respawn` is a warm
restart. `restart` is never "covered".

### Rules (rule set version 1), first match wins

| # | Rule | Reads | Group | Basis | Source |
|---|---|---|---|---|---|
| 1 | `path` (restart) | the engine's rules file | restart | declared | `INTERFACE.md`: startup, flags, env need a restart |
| 2 | `jit_kernel` | `@triton.jit`, `autotune`, `heuristics` | NATIVE | syntax | `kernel/jit.py`: the swap changes Python the launches never run |
| 3 | `dsl_kernel` | decorators the engine names | NATIVE | syntax | kernel finding `jit_target` |
| 4 | `constructor` | `__init__`, `__new__`, `__post_init__` | STATE | syntax | kernel finding `constructor_edit` |
| 5 | `compiled` | `@torch.compile`, `@torch.jit.script`, engine names | GRAPH | syntax | overview: code in a compiled region |
| 6 | `path` (other) | the engine's rules file | as declared | declared | the engine's adapter and `kb/` |
| 7 | `launch_flags` | a function `main`, or a body that calls `add_argument` | restart | name | `INTERFACE.md` |
| 8 | `name_resource` | names like `allocate_*`, `load_weights` | RESOURCE | name | overview: load-time allocation |
| 9 | `name_state` | names like `build_*`, `setup_*`, `create_*` | STATE | name | overview: objects built once at start |
| 10 | `default_op` | nothing matched | OP | default | overview: body of a per-call function |
| - | `module_level` | code outside every function | STATE, install `respawn` | syntax | kernel finding `outside_changes` |
| - | `native_file` | the file's extension | NATIVE | syntax | overview |

**Firm** means basis `syntax`: the text itself shows what the kernel or the lab keys on. All
other bases are heuristics, and the report gives their share as `uncertain`.

### The engine's rules file

JSON, kept with the engine. It declares, with a `why` for each line:

- `source_roots`, `native_roots`, `exclude`; `upstream` (`repo`, `commit`, `version`);
- `paths`: `{"glob", "group"}` declarations (`*` also crosses `/`);
- `decorators`: extra `dsl_kernel` and `compiled` names; `install_overrides` per rule;
- `capabilities`: for `swap`, `swap_recapture`, `jit_rebuild`, `rebuild`, `respawn`: the
  **evidence** (lab files that must exist, optionally matching a regular expression) and the
  upstream files the path does or does not reach (`reach`, `unreached`).

`covered` is true only when the evidence is found in the lab's files and the file is reached.
A rules file cannot claim a path the lab's tree does not hold.

### History

`history` classifies the upstream tree once, then reads the capabilities at each first-parent
commit of a ref that touched the lab's folder. The rules and the upstream tree stay fixed, so
the series shows only what the lab gained: coverage against commit index and commit date.

## Invariants

- Every function and every code line of a parsed file is in exactly one install class;
  `live_covered + respawn_covered + restart + not_covered = total`.
- Nothing is dropped silently: files that do not parse are counted in `unparsed`.
- The report holds no path outside the upstream tree and the repository.

## What it cannot tell

- **Reach.** Which functions a workload runs. A covered function may never run in the lab's
  workload; an uncovered one may not matter to it.
- **Captured or not.** Whether a plain function runs in Python or only inside a captured
  graph depends on the engine's configuration and call graph. Path declarations stand in for it.
- **Once or per call.** `default_op` assumes per call. A function that runs once at start
  swaps cleanly and proves nothing (`zero_witness`).
- **That an install works.** `covered` is not a receipt. Only a lab run with a witness shows
  an edit ran (AGENTS.md, Trust).
- **What an edit does.** A real edit can add a field or change a closure in an OP function
  and block (`new_fields`, `closure_changed`). The scan sees functions, not edits.

## Run it

From `core/livelab`. The fixture is a toy engine of 13 functions; its lab has a swap host
and a zygote, and no recapture or rebuild.

```sh
python3 -m survey scan --rules survey/fixture/rules.json --source survey/fixture/upstream \
    --repo survey/fixture --summary
```

```
toyengine: 13 functions, 47 lines; lab has swap, respawn
install class    functions  covered    lines  covered
swap                     3        2        8        6
swap_recapture           3        0        8        0
rebuild                  3        0        7        0
respawn                  1        1       11       11
restart                  3        -       13        -
functions: live 2, respawn 1, restart 3, no path yet 7; firm 4, uncertain 9 (69%)
lines: live 6, respawn 11, restart 13, no path yet 17; firm 20, uncertain 27 (57%)
native: 1 files, 4 lines (restart); unparsed: 1 files
```

Read it as: 2 of 3 swappable functions are covered (the third is in a file the lab does not
reach); the constructor, the two graph functions and the Triton kernel have no path in this
lab; 9 of 13 classifications are heuristics.

On an engine (JSON to a file, one line per function beside it), and its history:

```sh
python3 -m survey scan --rules <engine rules.json> --source <upstream tree> --repo ../.. \
    --commit origin/main --out <engine>.json --functions <engine>.functions.jsonl
python3 -m survey history --rules <engine rules.json> --source <upstream tree> --repo ../.. \
    --ref origin/main --out <engine>.history.json
```

## Golden tests

```sh
python3 -m unittest discover -s survey -t . -v     # CPU, under 1 s; the history case needs git
```

| Test | Pain point |
|---|---|
| `test_a_constructor_is_a_rebuild_never_a_covered_swap` | a constructor counted as live-editable on a lab that only swaps |
| `test_a_triton_kernel_needs_its_own_rebuild` | a JIT kernel counted as covered by a plain swap |
| `test_what_is_firm_and_what_is_a_guess` | a share that hides how much of it is guessed |
| `test_launch_flags_are_restart_by_path_and_by_body` | launch-flag code counted as swappable |
| `test_counts_add_up_and_nothing_is_dropped` | a function or file lost from the totals; a nested function counted twice |
| `test_coverage_comes_from_the_labs_files_not_from_the_rules` | a rules file that claims a path the lab does not have |
| `test_a_file_the_lab_does_not_reach_is_not_covered` | code of a process with no participant counted as covered |
| `test_coverage_steps_at_the_commit_that_adds_the_lab_file` | a history that moves for another reason than the lab's files |
| `test_the_command_line_writes_the_report_and_one_line_per_record` | a report that carries a local path |

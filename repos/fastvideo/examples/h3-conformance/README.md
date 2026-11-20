# Conformance c0-c10 on the real FastH3 lab

The template's conformance cases (`core/livelab/conformance/`) run against
`livelab/adapter/lab.py` hosting a real MiniMax H3 engine (FastH3-QAD-v2, the workload of
`kb/workload.md`) on GB200 GPUs, our 2026-11-07 runs. The cases drive the lab only through its
verbs; the edits are real H3 code (`adapter/patches.py`); a "fresh process" is a new FastVideo
process started from the edited files. Each check is one full 124-frame video.

## 2 GPUs (2 ranks, sequence parallel)

```
case                  result  seconds  evidence
c0_repeatable         pass     22.1    3 checks (one after reset ['cuda cache']) gave {'video': 'd0e55e892690e260', 'audio': '29b690c8ac7ae3be'}
c1_noop               pass      9.7    installed by ['R0 swap']; calls {'...minimax_h3_decoding:MiniMaxH3VideoDecodingStage.forward:109': [1, 1]}; output unchanged
c2_visible            pass      9.6    output {'video': 'd0e55e892690e260', ...} -> {'video': 'cd32f50b70d9479f', ...}
                                       calls per participant {'...MiniMaxH3VideoDecodingStage.forward:109': [1, 1]}
c3_fresh_equivalence  pass    253.2    visible: live == fresh == {'video': 'cd32f50b70d9479f', 'audio': '29b690c8ac7ae3be'}
                                       chain of 3: live == fresh == {'video': '23bd4a78af68a546', 'audio': '3d359a632084eff1'}
c4_uninstall          pass     19.3    uninstall ['txn-06219e55'] in 0.037977 s; identity 9c65a58963772d42 restored
c5_failed_install     pass     34.0    fail_apply: rolled_back, codes ['participant_failed'], base proven
                                       fail_commit: rolled_back, codes ['participant_failed'], base proven
                                       control: no fault, committed 3 targets, output changed
c6_refusal            pass     24.5    syntax: ['patch_invalid']; closure: ['closure_changed']
                                       module_state: ['empty_change', 'outside_changes']; empty: ['empty_change']
c7_chain              pass     38.6    d0... -> cd... -> 23... -> 23... -> undone one by one to base
c8_crash_and_hang     pass    610.3    crash: poisoned by ['participant_failed'], respawn 78.4 s
                                       hang: poisoned by ['participant_failed'], respawn 78.4 s
                                       lost: poisoned by ['participant_failed'], respawn 68.7 s
                                       tamper: poisoned by ['tampered'], respawn 48.4 s
c9_soak               pass    202.7    20 cycles; install median 0.0819 s, uninstall median 0.0355 s
c10_not_proven        pass    115.1    proven False, findings ['zero_witness', 'zero_witness']
                                       live {'video': 'd0e55e892690e260', ...} (base: True); fresh {'video': 'c7db7f61a884d310', ...}
```

11/11 on 2 ranks (c0-c7 and c10 in one lab session, c8 and c9 in a second one).

## 1 GPU (1 rank)

Our 2026-11-19 run (newer environment: torch 2.12, CUDA 13; base audio hash `ee49674663eae6e2`,
not the 11-07 golden's, `kb/lessons.md`), `python3 -m conformance --lab adapter.h3:H3Lab`:

```
case                  result  seconds  evidence
c0_repeatable         pass    186.7    3 checks (one after reset ['cuda cache']) gave {'video': 'd0e55e892690e260', 'audio': 'ee49674663eae6e2'}
c1_noop               pass     12.4    installed by ['R0 swap']; calls {'...MiniMaxH3VideoDecodingStage.forward:109': [1]}; output unchanged
c2_visible            pass     12.4    output {'video': 'd0e55e892690e260', ...} -> {'video': 'cd32f50b70d9479f', ...}
c3_fresh_equivalence  pass    154.3    visible: live == fresh == {'video': 'cd32f50b70d9479f', 'audio': 'ee49674663eae6e2'}
c4_uninstall          pass     24.7    uninstall in 0.01123 s; identity restored
c5_failed_install     pass     24.9    fail_apply and fail_commit: rolled_back, base proven; control committed 3 targets
c6_refusal            pass     31.2    patch_invalid / closure_changed / outside_changes / empty_change
c7_chain              pass     49.5    undone one by one to base
c8_crash_and_hang     pass    779.3    crash: poisoned by ['participant_failed'], respawn 22.2 s
                                       hang: poisoned by ['participant_failed'], respawn 53.5 s
                                       lost: poisoned by ['participant_failed'], respawn 23.0 s
                                       tamper: poisoned by ['tampered'], respawn 23.0 s
c9_soak               pass    259.4    20 cycles; install median 0.0335 s, uninstall median 0.0111 s
c10_not_proven        pass     56.9    proven False, findings ['zero_witness']
```

11/11 on 1 rank: c2 asks for a witness on every involved participant, and one rank is all of
them (it failed on 11-07, when c2 wanted at least two). c0's 186.7 s includes the first video's
FlashInfer build. The same day on 2 ranks: 11/11 again (c3 chain of 3 live == fresh
`23bd4a78af68a546`; c8 respawns 29.7-59.2 s; c9 install median 0.0624 s).

## 4 GPUs (4 ranks, SP4)

Our 2026-11-19 run via `python3 -m fastrsi conformance fastvideo --real -- --gpus 4` (108 receipts,
1989.1 s case sum, all 4 ranks witnessing every edit):

```
case                  result  seconds  evidence
c0_repeatable         pass     23.4    3 checks (one after reset ['cuda cache']) gave {'video': '0134242de77c3f4c', 'audio': '69356c2116d3c58a'}
c1_noop               pass      6.3    installed by ['R0 swap']; calls {'...MiniMaxH3VideoDecodingStage.forward:109': [1, 1, 1, 1]}; output unchanged
c2_visible            pass      6.3    output {'video': '0134242de77c3f4c', ...} -> {'video': '94849c72941908c3', ...}; calls [1, 1, 1, 1]
c3_fresh_equivalence  pass    176.0    visible: live == fresh == {'video': '94849c72941908c3', 'audio': '69356c2116d3c58a'}
                                       chain of 3: live == fresh == {'video': '837c08243df1ce30', 'audio': '35cbe7e342bfbbe0'}
c4_uninstall          pass     12.7    uninstall ['txn-6f3a9142'] in 0.032245 s; identity 163f6503796c0a2b restored
c5_failed_install     pass     12.9    fail_apply and fail_commit: rolled_back, base proven; control committed 3 targets
c6_refusal            pass     17.2    patch_invalid / closure_changed / outside_changes / empty_change
c7_chain              pass     25.0    01... -> 94... -> 83... -> 83... -> undone one by one to base
c8_crash_and_hang     pass   1514.9    crash: respawn 92.8 s; hang: respawn 89.0 s; lost: respawn 36.3 s; tamper: respawn 35.2 s
c9_soak               pass    131.4    20 cycles; install median 0.1140 s, uninstall median 0.0326 s
c10_not_proven        pass     63.0    proven False, findings ['zero_witness', 'zero_witness', 'zero_witness', 'zero_witness']
```

11/11 on 4 ranks (SP4), exiting cleanly to `0 MiB` on all 4 GPUs.

## 2 GPUs with `--zygote --holder` (SP2, targeted recovery & soak)

Our 2026-11-19 targeted run (`fastrsi conformance fastvideo --real --only c0_repeatable,c2_visible,c4_uninstall,c8_crash_and_hang,c9_soak -- --gpus 2 --zygote --holder`, `66` receipts, `5/5`, `0 MiB` after shutdown):

```
case               result  seconds  evidence
c0_repeatable      pass     25.732  3 checks (one after reset ['cuda cache']) gave {'video': 'd0e55e892690e260', 'audio': 'ee49674663eae6e2'}
c2_visible         pass      9.632  output {'video': 'd0e55e892690e260', ...} -> {'video': 'cd32f50b70d9479f', ...}; calls [1, 1]
c4_uninstall       pass     18.786  uninstall ['txn-d7e3dd2c'] in 0.016595 s; identity fbe225a4dfe5b535 restored
c8_crash_and_hang  pass   1362.848  crash: respawn 38.178275 s; hang: respawn 37.990799 s; lost: respawn 6.234872 s; tamper: respawn 5.726634 s
c9_soak            pass    197.869  20 cycles; install median 0.0589 s, uninstall median 0.0175 s
```

Every `c8` respawn mapped `text_encoder` (`0.145–0.358 s`), `vae` (`0.032–0.063 s`), and `transformer` (`0.185–0.738 s`) via holder IPC (`served: true`). `c8`'s `1362.8 s` total case wall includes two ~`600 s` default NCCL watchdog waits during the injected SP2 `rank1` crash and hang before `reset("respawn")` starts.

With asymmetric `SP >= 2` fault-beacon detection + parallel dirty `kill_tree(..., include_pg=False)` fast-fail (`4/4` pass, `0 MiB` across all 4 GPUs):

```
case               result  seconds  evidence
c0_repeatable      pass     25.529  3 checks (one after reset ['cuda cache']) gave {'video': 'd0e55e892690e260', 'audio': 'ee49674663eae6e2'}
c2_visible         pass      9.438  output {'video': 'd0e55e892690e260', ...} -> {'video': 'cd32f50b70d9479f', ...}; calls [1, 1]
c4_uninstall       pass     18.787  uninstall ['txn-a67f532d'] in 0.016656 s; identity 88d0e78248c36bf7 restored
c8_crash_and_hang  pass    218.459  crash: respawn 11.659885 s; hang: respawn 11.94372 s; lost: respawn 7.194716 s; tamper: respawn 6.079436 s
```

`c8` total wall fell **`6.24×` (`1362.848 s -> 218.459 s`)**; `text_encoder` (`0.142–0.323 s`) and `vae` (`0.031–0.035 s`) served via holder IPC on every respawn (`transformer` fell back cleanly to `1.66–1.70 s` local load + re-deposit after `RuntimeError: Broken pipe` on the two dirty-killed respawns, then served via holder IPC in `0.181–0.601 s` on `lost` and `tamper`).

## What these runs found (and the fixes they needed)

- **c8, hang:** the injected hang must outlast the check's deadline; the lab now sets it from
  `check_timeout` (`FastVideoLab.inject`).
- **c10:** a startup-only edit of H3's DMD rungs broke the fresh process (the precomputed AdaLN
  table covers only the trained ladder), so c10 now edits the NVFP4 VAE decoder's setup (its
  decode dtype), which runs once at load.
- **c5 on 2 ranks:** `_layout` runs only on the output rank, so an edit of it is (correctly) not
  proven on 2 ranks; c5's edit now changes three functions every rank runs.

## Not shown

Edits inside compiled or graph-captured code (the VAE is compiled), Triton kernel bodies edited
in place, and any respawn of a hung CUDA context (c8's hang sleeps in Python).

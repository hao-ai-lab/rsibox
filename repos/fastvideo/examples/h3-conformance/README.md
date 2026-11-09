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

```
case                  result  seconds  evidence
c0_repeatable         pass     27.8    3 checks (one after reset ['cuda cache']) gave {'video': 'd0e55e892690e260', 'audio': '29b690c8ac7ae3be'}
c1_noop               pass     12.7    installed by ['R0 swap']; calls {...: [1]}; output unchanged
c2_visible            FAIL     12.6    witness > 0 on every participant; got {'rank0': {...: 1}}   (needs >= 2 participants)
c3_fresh_equivalence  pass    265.8    visible and chain of 3: live == fresh
c4_uninstall          pass     25.2    uninstall in 0.014036 s; identity restored
c5_failed_install     pass     25.3    fail_apply and fail_commit: rolled_back, base proven; control committed 3 targets
c6_refusal            pass     31.8    patch_invalid / closure_changed / outside_changes / empty_change
c7_chain              pass     50.7    undone one by one to base
c8_crash_and_hang     pass    419.6    crash: poisoned by ['participant_failed'], respawn 42.8 s
                                       hang: poisoned by ['participant_failed'], respawn 98.6 s
                                       lost: poisoned by ['participant_failed'], respawn 36.8 s
                                       tamper: poisoned by ['tampered'], respawn 41.2 s
c9_soak               pass    266.8    20 cycles; install median 0.0523 s, uninstall median 0.0155 s
c10_not_proven        pass     89.2    proven False, findings ['zero_witness']; fresh video c7db7f61a884d310
```

c2 needs two participants by design (a witness on every rank); a 1-rank lab cannot show it.

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
in place, 4 ranks, and any respawn of a hung CUDA context (c8's hang sleeps in Python).

# Core live-lab runtime

The shared engine-agnostic runtime every engine's live lab builds on. No code
for one engine lives here: each engine's adapter and restart services are in
`repos/<engine>/livelab/`, which import the shared code from this folder (the engine packages
put it on `sys.path`; `FASTRSI_LIVELAB` overrides the folder). Onboarding scaffold for a new
engine: `template/`.

- `INTERFACE.md`: the fixed verbs and receipt format (version 0). `INTERFACE-v1.md` is the
  draft of version 1, on top of the transaction kernel; it is not adopted yet.
- `conformance/`: the cases c0-c10 every lab must pass before its receipts count:
  `python3 -m conformance --lab module:Class`. See `conformance/README.md`.
- `kernel/`: the transaction kernel (locate, journal, swap, witness, findings, two-phase
  coordinator, session, receipts), pure Python and importing no engine, with a toy engine and
  the toy lab (`kernel.toy_lab:ToyLab`) that passes c0-c10. See `kernel/README.md`.
- `fastrsi/`: shared stdlib runtime kit built on `kernel/` and `forksafe.py`: `BaseLab`
  (`LabUnderTest` base class), `RecaptureCoordinator`, `RoleEndpoint`, JSON IPC transports
  (`UnixJsonServer`, `FileChannelServer`, `MailboxServer`), cross-platform process-tree helpers
  (`proc.py`), stack-safe `MetaPathFinder` hooks & overlays (`hooks.py`), conformance harness
  helpers (`harness.py`), and the unified CLI (`python3 -m fastrsi` / `fastrsi`).
- `rsibox/`: a live lab as a service: the box daemon (any `LabUnderTest` behind HTTP with a
  bearer token), its client and CLI (`python3 -m rsibox` / `rsibox`), and the launchers that
  start it locally, in Docker, under gVisor or as a Slurm job. See `rsibox/README.md`.
- `ray_participants.py`: reach kernel participants inside `@ray.remote` actor copies and
  standalone files loaded without a `sys.modules` entry.
- `livestream.py`: automated cross-checking (`PricedBrief`, `cross_check_step_record`) and
  read-only run dashboard summaries (`build_dashboard_summary`).
- `survey/`: the static live-editability survey: classifies every function of an engine's
  upstream tree by function group and install class, and reads from the lab's files which of
  those paths the lab has; a history mode gives coverage per lab commit. See `survey/README.md`.
- `forksafe.py`: the pre-fork guards an engine zygote installs before its imports, so the
  processes it forks can still use CUDA: capability and current device from NVML
  (`install`), device properties from a probe process (`install_properties`), Triton's GPU
  target (`pin_triton_target`), CuTe DSL / `cuda-python` guards (`pin_cute_dsl_arch`,
  `guard_cuda_python`, `install_all_guards`), driver/bad-fork probes (`driver_initialized`,
  `poison_reasons`), and child unpatching (`unpatch`, `unpatch_child`).

From here (CPU, standard library only; torch cases skip without torch):

```sh
python3 -m unittest discover -s conformance -t . -v     # the cases fail labs that lack their property
python3 -m unittest discover -s kernel -t . -v          # kernel golden tests on the toy engine
python3 -m unittest discover -s fastrsi -t . -v         # stdlib runtime kit, BaseLab 11/11, CLI
python3 -m unittest discover -s . -p 'test_forksafe.py' -v
python3 -m unittest discover -s survey -t . -v          # survey golden tests on the fixture tree
python3 -m unittest discover -s rsibox -t . -v          # box daemon over real HTTP on the toy lab
python3 -m fastrsi list                                 # discover toy, template, and repo labs
python3 -m fastrsi conformance toy                      # c0-c10 on the toy lab
```

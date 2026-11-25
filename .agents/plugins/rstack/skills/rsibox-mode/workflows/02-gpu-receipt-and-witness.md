# SOP 2: GPU Verification & Receipt Honesty

1. **CPU Precondition Only**: Never declare GPU runtime readiness from CPU `ToyLab 11/11` or mocked tests.
2. **Safe GPU Lease**: Never use `pkill -f`; check idle hardware first, use PID/port registration, and release leases promptly.
3. **Multi-Shape Fresh Runs**: Run alternating fresh processes across `>= 2` distinct input shapes (`world_size >= 2` when distributed), separating cold compile/JIT time from warm steady state.
4. **Witness & Backlog Closure**: Verify `witness.py` records non-zero outputs and full capture+replay; if any GPU path was skipped, log it in `Not verified` and file/update a `status:needs-gpu` backlog issue (`rsibox-gpu-receipt`).

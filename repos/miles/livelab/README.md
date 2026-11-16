# `miles` Live Lab (`repos/miles/livelab/`)

Live code-swap, `code_epoch` rollout-buffer drain, and coupled trainer-to-rollout `S0` weight reset
adapter for `miles` (`radixark/miles`, `THUDM/slime` fork), built on `core/livelab/kernel/` and
`core/livelab/ray_participants.py`.

## Run Conformance (`11/11` on CPU, Spawned Rollout + Trainer Processes)

From `core/livelab`:

```sh
PYTHONPATH=../../repos/miles/livelab \
    python3 -m conformance --lab adapter.toy_lab:ToyMilesLab
```

## Run Golden Unit Tests

From the repository root:

```sh
PYTHONWARNINGS=error python3 -m unittest discover \
    -s repos/miles/livelab/adapter -t repos/miles/livelab -v
```

When `ray` is installed in the Python environment, `test_real_ray_actors_swap_and_undo_through_miles_step_gate`
also starts a local Ray cluster and verifies `@ray.remote` live swap, witness proof, and undo end
to end.

## Attaching to a Live `miles` / `slime` Ray Job

```python
from adapter.gate import MilesStepGate
from adapter.state import CoupledWeightReset, EpochDataBuffer
from kernel import session as SS

gate = MilesStepGate.from_controllers(
    rollout_source=rollout_executor,
    train_source=trainer_controller,
    mode="sync",  # or "async" with rollout_controller=inference_controller
    data_buffer=epoch_buffer,
)
live_session = SS.Session(gate.coordinator, ["miles.backends.training_utils.loss.objective"])
```

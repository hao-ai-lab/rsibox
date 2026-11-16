# Template Live Lab Skeleton (`template/livelab/`)

Runnable `LabUnderTest` scaffold for onboarding a new engine. Out of the box, `TemplateLab` loads
and closes cleanly while failing all `11/11` conformance cases (`c0`–`c10`) with step-by-step
`NotImplementedError` guidance.

## Run Conformance (`0/11` Expected on Unmodified Skeleton)

From `core/livelab`:

```sh
PYTHONPATH=../../template/livelab python3 -m conformance --lab adapter.lab:TemplateLab
```

## Run Skeleton Unit Tests

From the repository root:

```sh
python3 -m unittest discover -s template/livelab/adapter -t template/livelab -v
```

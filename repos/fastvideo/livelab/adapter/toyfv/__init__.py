"""A FastVideo-shaped toy engine for the adapter's CPU tests: spawned worker processes that
speak FastVideo's worker pipe protocol (worker/multiproc_executor.py: worker_busy_loop at the
pin), a pipeline of stages and a "transformer" with forward pre-hooks. No torch, no GPU."""

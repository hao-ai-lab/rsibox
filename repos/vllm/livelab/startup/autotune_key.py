"""Reuse FlashInfer autotune results across configs that run the same kernels.

vLLM stores autotune winners under a folder named by a hash of the whole vLLM config, so a change
that does not touch any tuned kernel (for example moving Engram tables from host to GPU memory)
still lands in a new, empty folder and tunes everything again. Setting FASTRSI_AUTOTUNE_KEY=<name>
makes every process (engine and weight-cache daemons) use the folder <name> instead.

Safe because FlashInfer keys each stored winner by operation and input shape inside the file:
entries for other shapes are ignored, and a shape it has never seen is tuned and added. Use one
key per model and parallel layout. Loaded by sitecustomize.py when the variable is set.
"""
import os

MODULE = "vllm.model_executor.warmup.flashinfer_autotune_cache"


def on_import(module):
    key = os.environ["FASTRSI_AUTOTUNE_KEY"]
    module.flashinfer_autotune_cache_hash = lambda runner: key

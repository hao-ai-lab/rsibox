"""Worker setup for the walkthrough's intentionally inactive sampler control."""
# Staging an edit can lazily import the target. Load the pristine legacy sampler
# here so its actual code objects can be compared before installation and after undo.
from vllm.v1.sample import sampler as _legacy_sampler  # noqa: F401

from adapter.worker_ext import KernelWorkerExtension as KernelWorkerExtension

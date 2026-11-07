# FastVideo lessons

Generic lessons from running the FastVideo live lab. Each row: what happened, what to do.

| Lesson | Do |
|---|---|
| FastVideo's `collective_rpc(timeout=...)` does not bound the wait in the multiprocess executor (its `recv` is unbounded, at the pin). A deadline that "times out" there leaves the late reply in the pipe, where the next call would read it as its own. | Poll each worker pipe with your own deadline; after a miss, treat the pipe as dirty until the worker group respawns (`livelab/adapter/lab.py: rank_call`). |
| A host script that builds a `VideoGenerator` at module level crashes with N >= 1 spawned workers: spawn re-imports the main module in every worker. | Keep all work under `if __name__ == "__main__":`. |
| On a shared network filesystem, a client that polls `exists(reply)` for a file another host creates can wait about 30 s (negative lookup cache). | Create the reply file empty before sending the request and poll its contents; each `open` revalidates (close-to-open). Measured: 30 s to 0.12 s per call. |
| The first video after a cold start spent 174 s of 218 s building FlashInfer NVFP4 kernels (`fp4_quantization`, `fp4_gemm_cutlass`) inside the text encoder stage. | Keep one FlashInfer workspace per node and reuse it; a new cache root is a cold start, not a baseline. |
| The FP4 attention path (`h3_vsa_fp4.enabled`) failed at the first forward on the pinned `fastvideo-kernel`: its Python calls an API the installed kernel package lacks. | Check that the installed `fastvideo-kernel` matches the source pin before choosing an attention path; we use BF16 VSA. |

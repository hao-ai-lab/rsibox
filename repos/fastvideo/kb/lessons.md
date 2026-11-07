# FastVideo lessons

Generic lessons from running the FastVideo live lab. Each row: what happened, what to do.

| Lesson | Do |
|---|---|
| FastVideo's `collective_rpc(timeout=...)` does not bound the wait in the multiprocess executor (its `recv` is unbounded, at the pin). A deadline that "times out" there leaves the late reply in the pipe, where the next call would read it as its own. | Poll each worker pipe with your own deadline; after a miss, treat the pipe as dirty until the worker group respawns (`livelab/adapter/lab.py: rank_call`). |
| A host script that builds a `VideoGenerator` at module level crashes with N >= 1 spawned workers: spawn re-imports the main module in every worker. | Keep all work under `if __name__ == "__main__":`. |
| On a shared network filesystem, a client that polls `exists(reply)` for a file another host creates can wait about 30 s (negative lookup cache). | Create the reply file empty before sending the request and poll its contents; each `open` revalidates (close-to-open). Measured: 30 s to 0.12 s per call. |
| The first video after a cold start spent 174 s of 218 s building FlashInfer NVFP4 kernels (`fp4_quantization`, `fp4_gemm_cutlass`) inside the text encoder stage. | Keep one FlashInfer workspace per node and reuse it; a new cache root is a cold start, not a baseline. |
| The FP4 attention path (`h3_vsa_fp4.enabled`) failed at the first forward on the pinned `fastvideo-kernel`: its Python calls an API the installed kernel package lacks. | Check that the installed `fastvideo-kernel` matches the source pin before choosing an attention path; we use BF16 VSA. |
| An edit that adds a `@triton.jit` kernel installed fine, then the next video crashed: Triton reads a kernel's source through `linecache`, which still held the file on disk, so it compiled the base file's lines. | The lab keeps `linecache` equal to the installed text of every module it changed, and puts the file back on undo (`livelab/adapter/agent.py`). Any decorator that reads source at definition time (Triton, Dynamo) needs this. |
| A new top-level `@triton.jit` kernel compiled and ran bit-exact, but its check read `proven: false`: a kernel launch runs no Python, so the witness on the kernel counts 0. | Build new kernels lazily in a Python function called on every dispatch (a cached builder): the witness then counts the dispatch. A launch counter for Triton targets is not in the kernel yet. |
| A conformance "startup-only" edit that changed H3's DMD rungs failed in the fresh process: H3 looks up AdaLN modulation in a table precomputed for the trained ladder only. | Pick startup-only edits that stay valid (we use the NVFP4 VAE decoder's decode dtype). |
| An edit inside a module-level `if HAVE_TRITON:` block was refused as `outside_changes`. | Define new functions at the top level; guard inside the function body. |


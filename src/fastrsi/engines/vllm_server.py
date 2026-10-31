"""Runs inside the container: one vLLM engine behind a Unix socket. Knows nothing about FastRSI."""

import hashlib
import json
import os
import socket
import sys
import time

from vllm import LLM, SamplingParams

TOPICS = ["a lighthouse", "the moon", "a robot", "rivers", "a library", "winter", "trains",
          "a garden", "the ocean", "a mountain", "music", "bridges", "a city at night", "bees",
          "a desert", "clocks"]


def main(path):
    llm = LLM(model="/model", tensor_parallel_size=1, seed=0, gpu_memory_utilization=0.3,
              max_num_seqs=64, max_model_len=2048, distributed_executor_backend="mp",
              compilation_config={"inductor_compile_config": {"benchmark_combo_kernel": False}})
    server = socket.socket(socket.AF_UNIX)
    server.bind(path + ".tmp")
    server.listen(8)
    os.rename(path + ".tmp", path)
    print("ready", flush=True)
    while True:
        connection, _ = server.accept()
        with connection:
            request = json.loads(connection.makefile().readline())
            if "reset" in request:
                llm.reset_prefix_cache()
                reply = {"digest": "0"}
            else:
                count, tokens = request.get("prompts", 32), request.get("tokens", 64)
                prompts = [f"Write a short paragraph about {TOPICS[i % len(TOPICS)]}, version {i}."
                           for i in range(count)]
                params = [SamplingParams(temperature=0.0, max_tokens=tokens, ignore_eos=True)
                          for _ in range(count)]
                began = time.perf_counter()
                outputs = llm.generate(prompts, params, use_tqdm=False)
                seconds = time.perf_counter() - began
                ids = [list(output.outputs[0].token_ids) for output in outputs]
                digest = hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16]
                reply = {"digest": digest, "engine_seconds": seconds,
                         "tokens": sum(map(len, ids)), "text": outputs[0].outputs[0].text[:60]}
            connection.sendall(json.dumps(reply).encode() + b"\n")


if __name__ == "__main__":
    main(sys.argv[1])

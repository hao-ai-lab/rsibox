"""Runs inside the container: one SGLang engine behind a Unix socket."""

import hashlib
import json
import os
import socket
import sys
import time

import sglang

TOPICS = ["a lighthouse", "the moon", "a robot", "rivers", "a library", "winter", "trains",
          "a garden", "the ocean", "a mountain", "music", "bridges", "a city at night", "bees",
          "a desert", "clocks"]


def main(path):
    engine = sglang.Engine(model_path="/model", mem_fraction_static=0.3, random_seed=0,
                           log_level="error", enable_metrics=False, max_running_requests=1)
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
                engine.flush_cache()
                reply = {"digest": "0"}
            else:
                count, tokens = request.get("prompts", 32), request.get("tokens", 64)
                prompts = [f"Write a short paragraph about {TOPICS[i % len(TOPICS)]}, version {i}."
                           for i in range(count)]
                params = {"temperature": 0.0, "max_new_tokens": tokens, "ignore_eos": True}
                began = time.perf_counter()
                outputs = engine.generate(prompt=prompts, sampling_params=params)
                seconds = time.perf_counter() - began
                ids = [output.get("output_ids") or output["text"] for output in outputs]
                digest = hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16]
                reply = {"digest": digest, "engine_seconds": seconds,
                         "text": outputs[0]["text"][:60]}
            connection.sendall(json.dumps(reply).encode() + b"\n")


if __name__ == "__main__":
    main(sys.argv[1])

"""A small multi-process engine: one driver and two workers, standard library only."""

import hashlib
import json
import multiprocessing
import os
import socket
import sys

import toymodel


def worker(pipe):
    accumulator = toymodel.Accumulator()
    while True:
        message = pipe.recv()
        if message == "reset":
            accumulator = toymodel.Accumulator()
            pipe.send(0)
        else:
            pipe.send([accumulator.add(toymodel.forward(x)) for x in message])


def main(path):
    context = multiprocessing.get_context("spawn")
    pipes = []
    for _ in range(2):
        here, there = context.Pipe()
        context.Process(target=worker, args=(there,), daemon=True).start()
        pipes.append(here)
    server = socket.socket(socket.AF_UNIX)
    server.bind(path)
    server.listen(8)
    print("ready", flush=True)
    while True:
        connection, _ = server.accept()
        with connection:
            request = json.loads(connection.makefile().readline())
            if "reset" in request:
                for pipe in pipes:
                    pipe.send("reset")
                    pipe.recv()
                reply = {"digest": "0", "pids": [os.getpid()]}
            else:
                for pipe in pipes:
                    pipe.send(request["xs"])
                values = [toymodel.postprocess(pipe.recv()) for pipe in pipes]
                digest = hashlib.sha256(json.dumps(values).encode()).hexdigest()[:16]
                reply = {"digest": digest, "values": values}
            connection.sendall(json.dumps(reply).encode() + b"\n")


if __name__ == "__main__":
    main(sys.argv[1])

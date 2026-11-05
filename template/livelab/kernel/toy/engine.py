"""Toy engine: one step = shift(activate(scale(x))), then the token callbacks."""
import threading
import time

from .ops import activate, make_shift, scale


class Engine:
    def __init__(self, delay=0.0):
        self.steps = 0
        self.tokens = []
        self.delay = delay
        self.shift = make_shift(10)
        self.callbacks = [self.on_token]
        self.looping = False
        self._boundary = []

    def on_token(self, t):
        self.tokens.append(t)

    def step(self, x):
        y = self.shift(activate(scale(x)))
        for cb in self.callbacks:
            cb(y)
        self.steps += 1
        return y

    def drain(self):
        self.tokens.clear()

    def run_boundary(self):
        while self._boundary:
            self._boundary.pop(0)()

    def at_boundary(self, fn, timeout=5.0):
        """Run fn between two steps (the quiesce point) and return its result."""
        if not self.looping:
            return fn()
        box, done = {}, threading.Event()

        def job():
            try:
                box["value"] = fn()
            except BaseException as e:
                box["error"] = e
            done.set()
        self._boundary.append(job)
        if not done.wait(timeout):
            raise TimeoutError(f"no step boundary within {timeout} s")
        if "error" in box:
            raise box["error"]
        return box["value"]


def run_loop(engine, stop, x=1):
    engine.looping = True
    try:
        while not stop.is_set():
            engine.run_boundary()
            engine.step(x)
            time.sleep(engine.delay)
    finally:
        engine.looping = False
        engine.run_boundary()

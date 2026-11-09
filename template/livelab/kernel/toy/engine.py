"""Toy engine: one step = shift(activate(scale(x))), then the token callbacks."""
import threading
import time

from .ops import activate, make_shift, scale


class Engine:
    def __init__(self, delay=0.0):
        self.steps = 0
        self.tokens = []
        self.ran = []
        self.cache = {}
        self.delay = delay
        self.shift = make_shift(10)
        self.callbacks = [self.on_token]
        self.boundary = Gap()

    def on_token(self, t):
        self.tokens.append(t)

    def step(self, x):
        self.ran.append("base")
        y = self.shift(activate(scale(x)))
        for cb in self.callbacks:
            cb(y)
        self.steps += 1
        return y

    def drain(self):
        self.tokens.clear()

    @property
    def looping(self):
        return self.boundary.active

    def run_boundary(self):
        self.boundary.drain()

    def at_boundary(self, fn, timeout=5.0):
        """Run fn between two of this engine's steps (its quiesce point); return its result."""
        return self.boundary.call(fn, timeout)


class Gap:
    """Work queued for the gap between two steps of a loop. `call` runs fn now when no loop
    is active, else queues it and waits for the loop's next `drain`."""

    def __init__(self):
        self.active = False
        self._queue = []

    def drain(self):
        while self._queue:
            self._queue.pop(0)()

    def call(self, fn, timeout=5.0):
        if not self.active:
            return fn()
        box, done = {}, threading.Event()

        def job():
            try:
                box["value"] = fn()
            except BaseException as e:
                box["error"] = e
            done.set()
        self._queue.append(job)
        if not done.wait(timeout):
            raise TimeoutError(f"no step gap within {timeout} s")
        if "error" in box:
            raise box["error"]
        return box["value"]


def run_loop(engine, stop, x=1):
    engine.boundary.active = True
    try:
        while not stop.is_set():
            engine.run_boundary()
            engine.step(x)
            time.sleep(engine.delay)
    finally:
        engine.boundary.active = False
        engine.run_boundary()


class Cluster:
    """The engine core of a multi-rank toy: it owns the step loop. One cluster step runs every
    rank's step (each at its own boundary first) and logs which code version each rank ran.
    Work queued with `at_gap` runs between two cluster steps, never during one."""

    def __init__(self, engines, delay=0.0005):
        self.engines = dict(engines)       # rank name -> Engine
        self.delay = delay
        self.log = []                      # per cluster step: the version each rank's step ran
        self.ends = []                     # per cluster step: time.perf_counter() at its end
        self.gap = Gap()

    def step(self, x=1):
        for e in self.engines.values():
            e.run_boundary()
            e.step(x)
        self.log.append(tuple(e.ran[-1] for e in self.engines.values()))
        self.ends.append(time.perf_counter())

    def longest_stall(self):
        """Longest time between the ends of two consecutive steps (seconds)."""
        return max((b - a for a, b in zip(self.ends, self.ends[1:])), default=0.0)

    def at_gap(self, fn, timeout=5.0):
        return self.gap.call(fn, timeout)

    def run(self, stop, x=1):
        self.gap.active = True
        for e in self.engines.values():
            e.boundary.active = True
        try:
            while not stop.is_set():
                self.gap.drain()
                self.step(x)
                time.sleep(self.delay)
        finally:
            self.gap.active = False
            for e in self.engines.values():
                e.boundary.active = False
                e.run_boundary()
            self.gap.drain()

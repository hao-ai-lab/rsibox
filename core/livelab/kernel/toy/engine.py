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
        self._lock = threading.Lock()

    def start(self):
        with self._lock:
            self.active = True

    def stop(self):
        with self._lock:
            self.active = False
        self.drain()

    def drain(self):
        while True:
            with self._lock:
                if not self._queue:
                    return
                queued_job = self._queue.pop(0)
            queued_job()

    def call(self, fn, timeout=5.0):
        result_box, job_done = {}, threading.Event()

        def job():
            try:
                result_box["value"] = fn()
            except BaseException as error:
                result_box["error"] = error
            job_done.set()

        with self._lock:
            if not self.active:
                run_immediately = True
            else:
                run_immediately = False
                self._queue.append(job)
        if run_immediately:
            return fn()
        if not self.active:
            self.drain()
        if not job_done.wait(timeout):
            raise TimeoutError(f"no step gap within {timeout} s")
        if "error" in result_box:
            raise result_box["error"]
        return result_box["value"]


def run_loop(engine, stop, x=1):
    engine.boundary.start()
    try:
        while not stop.is_set():
            engine.run_boundary()
            engine.step(x)
            time.sleep(engine.delay)
    finally:
        engine.boundary.stop()


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
        for engine in self.engines.values():
            engine.run_boundary()
            engine.step(x)
        self.log.append(tuple(engine.ran[-1] for engine in self.engines.values()))
        self.ends.append(time.perf_counter())

    def longest_stall(self):
        """Longest time between the ends of two consecutive steps (seconds)."""
        return max((end_b - end_a for end_a, end_b in zip(self.ends, self.ends[1:])), default=0.0)

    def at_gap(self, fn, timeout=5.0):
        return self.gap.call(fn, timeout)

    def run(self, stop, x=1):
        self.gap.start()
        for engine in self.engines.values():
            engine.boundary.start()
        try:
            while not stop.is_set():
                self.gap.drain()
                self.step(x)
                time.sleep(self.delay)
        finally:
            for engine in self.engines.values():
                engine.boundary.stop()
            self.gap.stop()

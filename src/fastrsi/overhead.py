"""What FastRSI itself costs, per engine. Every number here is measured in this run."""

import json
import os
import shutil
import statistics
import time

from fastrsi import edits
from fastrsi.core import Session, call


def summary(values):
    ordered = sorted(values)
    return {"n": len(ordered), "median": statistics.median(ordered), "min": ordered[0],
            "max": ordered[-1], "p90": ordered[int(0.9 * (len(ordered) - 1))]}


def timed_runs(engine, spec, repeats, warmup):
    for _ in range(warmup):
        engine.exercise(timeout=spec.exercise_timeout, **spec.exercise_args)
    return [engine.exercise(timeout=spec.exercise_timeout, **spec.exercise_args)["seconds"]
            for _ in range(repeats)]


def measure_stock_vs_agent(engine_class, workdir, options, rounds, repeats, warmup):
    """O1 and O3. Alternate arms so drift over time does not favor either."""
    arms = {"stock": [], "agent": []}
    memory = {"stock": [], "agent": []}
    startup = {"stock": [], "agent": []}
    order = []
    for index in range(rounds * 2):
        arm = ("stock", "agent")[(index + index // 2) % 2]
        engine = engine_class(os.path.join(workdir, f"{arm}-{index}"), **options)
        began = time.perf_counter()
        engine.start(agent=arm == "agent")
        startup[arm].append(time.perf_counter() - began)
        try:
            spec = engine.spec()
            runs = timed_runs(engine, spec, repeats, warmup)
            arms[arm] += runs
            memory[arm].append(engine.stats())
            order.append({"arm": arm, "median": statistics.median(runs), "max": max(runs),
                          "startup_s": startup[arm][-1], **memory[arm][-1]})
        finally:
            engine.stop()
    medians = {arm: statistics.median(values) for arm, values in arms.items()}
    return {
        "exercise_s": {arm: summary(values) for arm, values in arms.items()},
        "idle_cost_pct": 100 * (medians["agent"] / medians["stock"] - 1),
        "startup_s": {arm: summary(values) for arm, values in startup.items()},
        "memory": memory,
        "rounds": order,
    }


def measure_session(engine_class, workdir, options, cycles):
    """O2, O4, O5, O6, O8, O9 on one resident engine."""
    engine = engine_class(os.path.join(workdir, "session-engine"), **options)
    engine.start()
    session = Session(engine, os.path.join(workdir, "session"), max_installs=cycles * 3)
    results = {}
    try:
        spec = engine.spec()
        path, old, new = spec.visible
        edit = edits.replace(engine.root(), path, old, new)
        for _ in range(3):
            session.exercise(timeout=spec.exercise_timeout, **spec.exercise_args)
        results["base"] = timed_runs(engine, spec, 5, 0)

        installs, undos, reads, stats = [], [], [], [engine.stats()]
        for _ in range(cycles):
            receipt = session.install(edit)
            installs.append(receipt["seconds"])
            session.exercise(timeout=spec.exercise_timeout, **spec.exercise_args)
            undos.append(session.undo()["seconds"])
            reads.append(session.exercise(timeout=spec.exercise_timeout,
                                          **spec.exercise_args)["seconds"])
            stats.append(engine.stats())
        results["install_s"] = summary(installs)
        results["undo_s"] = summary(undos)
        results["growth"] = {"first": stats[0], "last": stats[-1], "cycles": cycles}
        quarter = max(1, cycles // 4)
        early, late = statistics.median(reads[:quarter]), statistics.median(reads[-quarter:])
        results["base_read_drift"] = {"early_s": early, "late_s": late,
                                      "percent": 100 * (late / early - 1), "cycles": cycles}

        began = time.perf_counter()
        engine.reset()
        results["reset_s"] = time.perf_counter() - began

        noop_path, noop_old, noop_new = spec.noop
        noop = edits.replace(engine.root(), noop_path, noop_old, noop_new)
        for mode in ("first", "count"):
            session.install(noop, witness=mode)
            results[f"witness_{mode}_s"] = summary(timed_runs(engine, spec, 5, 1))
            session.undo()

        hang = edits.replace(engine.root(), *spec.hang)
        session.install(hang)
        began = time.perf_counter()
        session.exercise(timeout=spec.exercise_timeout, **spec.exercise_args)
        results["detect_hang_s"] = time.perf_counter() - began
        began = time.perf_counter()
        session.restart()
        results["recover_s"] = time.perf_counter() - began
        results["agent_ping_s"] = _ping(session)
    finally:
        engine.stop()
    return results


def _ping(session):
    socket_path = session.sockets()[0]
    samples = []
    for _ in range(200):
        began = time.perf_counter()
        call(socket_path, "ping")
        samples.append(time.perf_counter() - began)
    return summary(samples)


def run(engine_class, workdir, options=None, rounds=3, repeats=20, warmup=3, cycles=20):
    options = options or {}
    workdir = os.path.abspath(workdir)
    shutil.rmtree(workdir, ignore_errors=True)
    os.makedirs(workdir)
    record = {"engine": engine_class.name, "time": time.time(), "rounds": rounds,
              "repeats": repeats, "cycles": cycles}
    record["stock_vs_agent"] = measure_stock_vs_agent(
        engine_class, workdir, options, rounds, repeats, warmup)
    record["session"] = measure_session(engine_class, workdir, options, cycles)
    with open(os.path.join(workdir, "overhead.json"), "w") as handle:
        json.dump(record, handle, indent=1)
    return record

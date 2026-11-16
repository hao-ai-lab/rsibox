"""Step-gap adapter across `miles` (`radixark/miles` / `THUDM/slime`) Ray actors."""
import contextlib, importlib, os, sys
import adapter  # noqa: F401
from kernel import coordinator as C, locate as L
import ray_participants as RP


def serve_actor(host, name, method, payload):
    """Actor-side wrapper over `ray_participants.serve`: imports a by-value actor's defining module
    if cloudpickle skipped it, primes `#actor` views, and scopes `fingerprint` to loaded modules."""
    own_mod = getattr(getattr(type(host), "__ray_actor_class__", None), "__module__", None)
    if own_mod and own_mod not in sys.modules:
        importlib.import_module(own_mod)
    local = RP._LOCAL.get(name) or RP._LOCAL.setdefault(name, C.LocalParticipant(name))
    if (view := RP.actor_view(host)) is not None:
        local.alias[view[0] + RP.ACTOR] = view[1]
    if method == "fingerprint":
        scoped = []
        for entry in payload:
            if entry.endswith(".py") or os.sep in entry:
                actual, loaded_mod = RP.file_namespace(entry)
                if loaded_mod is not None:
                    local.alias[key := RP.file_module(entry)] = actual; scoped.append(key)
            elif entry in sys.modules or entry in local.alias:
                scoped.append(entry)
        payload = scoped
    return RP.serve(host, name, method, payload)


class MilesActorParticipant(RP.RayActorParticipant):
    """`RayActorParticipant` dispatching through `serve_actor` for multi-role Ray sessions."""
    def __init__(self, name, handle, get=None, timeout_s=None, bounded=False):
        fetch = get or RP._ray_get
        C.CallParticipant.__init__(
            self, name, lambda m, p: fetch(handle.__ray_call__.remote(serve_actor, name, m, p), timeout_s), bounded)


def discover_actors(rollout_source=None, train_source=None):
    """Extract `(rollout_handles, train_handles)` from `miles` or `slime` controllers."""
    def _extract(source, prefix):
        if source is None: return {}
        if isinstance(source, dict): return {str(k): v for k, v in source.items()}
        for attr in ("_actor_handlers", "actors", "_workers", "workers"):
            if isinstance(handles := getattr(source, attr, None), (list, tuple)):
                return {f"{prefix}{i}": h for i, h in enumerate(handles)}
        if isinstance(cells := getattr(source, "_cells", None), (list, tuple)):
            if found := {f"{prefix}_{c.cell_id}_{i}": h for c in cells for i, h in enumerate(getattr(c, "workers", ()) or ())}:
                return found
        if isinstance(source, (list, tuple)): return {f"{prefix}{i}": h for i, h in enumerate(source)}
        return {f"{prefix}0": source}
    return _extract(rollout_source, "rollout"), _extract(train_source, "trainer")


def clean_witness(raw_witness, proof_ok):
    """Strip dead `@ray.remote` handle keys when `#actor` ran; omit abstaining roles (`{}`)."""
    cleaned = {}
    for part_name, counts in (raw_witness or {}).items():
        actor_keys = {str(k).split(RP.ACTOR + ":", 1)[1] for k in counts if RP.ACTOR + ":" in str(k)}
        filtered = {k: n for k, n in counts.items()
                    if not (RP.ACTOR not in k and ":" in k and n == 0 and k.split(":", 1)[1] in actor_keys)}
        if filtered or not proof_ok: cleaned[part_name] = filtered
    return cleaned


class MilesStepGate:
    """Host-side step-gap gate over rollout and trainer `MilesActorParticipant`s."""
    def __init__(self, rollout_actors, train_actors, *, extra_actors=None, get=None,
                 timeout_s=60.0, mode="sync", rollout_controller=None, data_buffer=None,
                 at_gap_fn=None, writer=None, engine_meta=None):
        if mode not in ("sync", "async"): raise ValueError(f"unsupported mode {mode!r}")
        self.mode, self.timeout_s, self.fetch = mode, timeout_s, get or RP._ray_get
        self.rollout_controller, self.data_buffer, self._external_gap = rollout_controller, data_buffer, at_gap_fn
        self.handles = {**rollout_actors, **train_actors, **(extra_actors or {})}
        self.participants = [MilesActorParticipant(n, h, get=self.fetch, timeout_s=timeout_s)
                             for n, h in self.handles.items()]
        meta = engine_meta or {"name": "miles", "version": "0.2.0", "commit": None}
        self.coordinator = C.StepGapCoordinator(
            self.participants, self.at_gap, writer=writer, interface_version="1-draft",
            engine=meta, phase2_deadline_s=timeout_s)

    @classmethod
    def from_controllers(cls, rollout_source=None, train_source=None, **kwargs):
        rollout_map, train_map = discover_actors(rollout_source, train_source)
        return cls(rollout_map, train_map, **kwargs)

    @contextlib.contextmanager
    def _quiesce_async(self):
        if self.mode != "async": yield; return
        if self.data_buffer is not None: self.data_buffer.pause()
        for attr in ("pause_generation", "pause", "flush_cache"):
            if callable(fn := getattr(self.rollout_controller, attr, None)): fn()
        try: yield
        finally:
            ctl = self.rollout_controller
            if callable(resume := getattr(ctl, "continue_generation", None) or getattr(ctl, "resume", None)): resume()
            if self.data_buffer is not None: self.data_buffer.resume()

    def at_gap(self, fn, timeout=None):
        def guarded():
            with self._quiesce_async(): return fn()
        return self._external_gap(guarded, timeout or self.timeout_s) if self._external_gap else guarded()

    def build_plan(self, patch, base_text):
        modules = {}
        for target_key, source_text in (patch.get("files") or {}).items():
            is_path = target_key.endswith(".py") or os.sep in target_key
            module_key = RP.file_module(target_key) if is_path else target_key
            spec = {"file": target_key, "source": source_text} if is_path else {"source": source_text}
            if target_key in base_text: spec["base_hash"] = L.text_hash(base_text[target_key])
            modules[module_key] = spec
        plan = {"modules": modules, "accept": list(patch.get("accept", [])),
                "undo_resets": list(patch.get("undo_resets", [])), "recipes": ["R0 swap"]}
        if missing := RP.unloaded(self.participants, plan):
            return None, [{"check": "unloaded_module", "observed": f"no actor loaded {missing}",
                           "expected": "every edited module or file is loaded by at least one actor"}]
        return plan, []

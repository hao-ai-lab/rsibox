"""Rollout `@ray.remote` actor (`AgentLoopWorkerTQ` analog) and standalone `@ray.remote` stand-in."""
import sys
import types

from adapter.toyverl import algos


def remote_actor(cls):
    """Mirror `@ray.remote`: replace `cls` in its module with an `ActorClass` handle whose
    `__ray_metadata__.modified_class.__ray_actor_class__` holds `cls`."""
    modified = type(
        cls.__name__,
        (cls,),
        {"__ray_actor_class__": cls, "__module__": cls.__module__},
    )
    handle = type(f"ActorClass({cls.__name__})", (), {})()
    handle.__dict__["__ray_metadata__"] = types.SimpleNamespace(modified_class=modified)
    return handle


def instantiate_actor_copy(handle, *args, **kwargs):
    """Instantiate a by-value deserialized copy of `handle`'s class (as Ray cloudpickle does)."""
    original = vars(handle)["__ray_metadata__"].modified_class.__ray_actor_class__
    mod_dict = vars(sys.modules[original.__module__])
    referenced = set()

    def scan_code(code_obj):
        referenced.update(code_obj.co_names)
        for const in code_obj.co_consts:
            if isinstance(const, types.CodeType):
                scan_code(const)

    actor_globals = {
        key: mod_dict[key]
        for key in ("__name__", "__file__", "__package__")
        if key in mod_dict
    }
    body = {}
    for attr_name, value in vars(original).items():
        if attr_name in ("__dict__", "__weakref__"):
            continue
        if isinstance(value, types.FunctionType):
            scan_code(value.__code__)
            copied = types.FunctionType(
                value.__code__,
                actor_globals,
                value.__name__,
                value.__defaults__,
                value.__closure__,
            )
            copied.__qualname__ = value.__qualname__
            copied.__kwdefaults__ = value.__kwdefaults__
            value = copied
        body[attr_name] = value
    actor_globals.update({name: mod_dict[name] for name in referenced if name in mod_dict})
    copied_cls = type(original.__name__, original.__bases__, body)
    runtime_cls = type(
        copied_cls.__name__,
        (copied_cls,),
        {"__ray_actor_class__": copied_cls, "__module__": copied_cls.__module__},
    )
    return runtime_cls(*args, **kwargs)


def sample_token(prompt_id, weight_sum, bias):
    raw = int(round((prompt_id * 7 + weight_sum * 100.0 + bias * 10.0))) % 29
    return algos.check_tokens([raw + 1])[0]


@remote_actor
class RolloutActor:
    def __init__(self, rank=0):
        self.rank = rank
        self.bias = algos.init_schedule()
        self.weights = [1.0, 0.5, -0.25]
        self.kv_cache = {}

    def sync_weights(self, new_weights):
        self.weights = list(new_weights)
        self.kv_cache.clear()
        return len(self.weights)

    def generate(self, prompts):
        weight_sum = sum(self.weights)
        outputs = []
        for prompt_id in prompts:
            cached = self.kv_cache.get(prompt_id)
            if cached is None:
                tok = sample_token(prompt_id, weight_sum, self.bias)
                self.kv_cache[prompt_id] = tok
            else:
                tok = cached
            outputs.append(tok)
        return outputs

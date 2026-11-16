"""Stand-in `@ray.remote` handle and cloudpickle-by-value actor instantiation."""
from kernel.test_ray_participants import by_value, fake_remote as remote  # noqa: F401


def instantiate_actor(handle, **kwargs):
    original = vars(handle)["__ray_metadata__"].modified_class.__ray_actor_class__
    copied = by_value(original)
    actor_cls = type(copied.__name__, (copied,), {"__ray_actor_class__": copied, "__module__": copied.__module__})
    return actor_cls(**kwargs)

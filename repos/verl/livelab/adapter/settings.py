"""Live performance settings (`--set KEY=VALUE`): change fields verl reads on every step."""

LIVE = {
    "actor.ppo_micro_batch_size_per_gpu": ("actor", "micro_batch_size_per_gpu", "train"),
    "rollout.log_prob_micro_batch_size_per_gpu": ("actor", "infer_micro_batch_size_per_gpu", "infer"),
    "ref.log_prob_micro_batch_size_per_gpu": ("ref", "infer_micro_batch_size_per_gpu", "infer"),
    "actor.use_dynamic_bsz": ("actor", "use_dynamic_bsz", "bool"),
    "ref.log_prob_use_dynamic_bsz": ("ref", "use_dynamic_bsz", "bool"),
    "actor.ppo_max_token_len_per_gpu": ("actor", "max_token_len_per_gpu", "tokens"),
    "rollout.log_prob_max_token_len_per_gpu": ("actor", "infer_max_token_len_per_gpu", "tokens"),
    "ref.log_prob_max_token_len_per_gpu": ("ref", "infer_max_token_len_per_gpu", "tokens"),
}
ROLE = "actor_rollout_ref"


def plan(requested, batch_sizes):
    validated = {}
    for key, text in (requested or {}).items():
        if key not in LIVE:
            raise ValueError(f"{key}: not a live setting (read at startup or changes what is learned); restart needed")
        kind = LIVE[key][2]
        if kind == "bool":
            low = str(text).strip().lower()
            if low not in ("true", "false"):
                raise ValueError(f"{key}: expected true or false, got {text!r}")
            val = low == "true"
        else:
            try:
                val = int(text)
            except ValueError:
                raise ValueError(f"{key}: expected a positive integer, got {text!r}") from None
            if val <= 0:
                raise ValueError(f"{key}: expected a positive integer, got {text!r}")
            if kind in ("train", "infer") and batch_sizes[kind] % val:
                raise ValueError(f"{key}={val} does not divide the per-GPU batch of {batch_sizes[kind]}")
        validated[key] = val
    return validated


def apply(host, values):
    worker, previous = host.worker_dict[ROLE], {}
    for key, val in values.items():
        attr, field, _ = LIVE[key]
        cfg = getattr(worker, attr).engine_config
        previous[key] = getattr(cfg, field)
        setattr(cfg, field, val)
    return previous

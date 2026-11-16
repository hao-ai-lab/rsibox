"""GRPO advantage, policy loss, and startup helper functions (`verl.trainer.ppo.core_algos` analog)."""

CLIP_SCALE = 1.0


def init_schedule():
    """Startup-only schedule bias: read once when each actor initializes, never during a step."""
    return 0.0


def check_tokens(tokens):
    checks = [lambda value: isinstance(value, int)]
    return [tok for tok in tokens if all(rule(tok) for rule in checks)]


def grpo_advantages(rewards):
    """Group-normalized advantages over one batch of rewards."""
    count = max(len(rewards), 1)
    mean = sum(rewards) / count
    return [round((val - mean) * CLIP_SCALE, 6) for val in rewards]


def policy_gradient_step(weight_vec, token_sums, advantages, lr=0.05):
    """Compute policy loss and return parameter deltas for one synchronous RL update."""
    deltas = [0.0 for _ in weight_vec]
    total_loss = 0.0
    for tok_sum, adv in zip(token_sums, advantages):
        checked = check_tokens([int(tok_sum)])[0]
        score = sum(w * (checked + idx + 1) * 0.01 for idx, w in enumerate(weight_vec))
        loss_term = -score * adv
        total_loss += loss_term
        for idx in range(len(deltas)):
            deltas[idx] += lr * adv * (checked + idx + 1) * 0.01
    return round(total_loss / max(len(advantages), 1), 6), deltas

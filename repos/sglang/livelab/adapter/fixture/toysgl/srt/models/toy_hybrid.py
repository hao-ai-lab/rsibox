"""A hybrid toy model: decoder layers (graph-capturable), attention over the request's KV,
and one recurrent (Mamba/GDN-like) state per request slot."""
from adapter.fixture.toysgl.srt.layers import activation, layernorm

MODULUS = layernorm.MODULUS


class ToyHybridForCausalLM:
    num_layers = 2

    def embed(self, token, position):
        return (token * 131 + position * 17 + 1) % MODULUS

    def layers_forward(self, h):
        """The decoder layers: what a decode graph captures."""
        for _ in range(self.num_layers):
            h = activation.silu_and_mul(layernorm.rms_norm(h))
        return h

    def attend(self, h, kv_sum):
        return (h + 3 * kv_sum) % MODULUS

    def mamba_step(self, state, out):
        return (state * 5 + out) % MODULUS

    def final(self, out, state):
        return (out + state) % MODULUS

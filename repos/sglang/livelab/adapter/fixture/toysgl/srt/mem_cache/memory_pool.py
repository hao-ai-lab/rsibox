"""Request, KV and recurrent-state pools, and the prefix cache (sglang/srt/mem_cache/)."""


class ReqToTokenPool:
    def __init__(self, size):
        self.size = size
        self.free_slots = list(range(size))

    def alloc(self):
        return self.free_slots.pop(0) if self.free_slots else None

    def free(self, slot):
        self.free_slots.append(slot)

    def available_size(self):
        return len(self.free_slots)

    def clear(self):
        self.free_slots = list(range(self.size))


class TokenToKVPoolAllocator:
    def __init__(self, size):
        self.size = size
        self.used = 0

    def alloc(self, n):
        if self.used + n > self.size:
            return False
        self.used += n
        return True

    def free(self, n):
        self.used -= n

    def available_size(self):
        return self.size - self.used

    def clear(self):
        self.used = 0


class MambaPool:
    """One recurrent state per slot. A slot is zeroed when it is handed out: a slot reused
    without that reset carries another request's state (SGLang issue 50139)."""

    def __init__(self, size):
        self.size = size
        self.states = [0] * size
        self.free_slots = list(range(size))

    def alloc(self):
        if not self.free_slots:
            return None
        slot = self.free_slots.pop(0)
        self.states[slot] = 0
        return slot

    def free(self, slot):
        self.free_slots.append(slot)

    def available_size(self):
        return len(self.free_slots)

    def clear(self):
        self.states = [0] * self.size
        self.free_slots = list(range(self.size))


class RadixCache:
    """Finished sequences -> per-position KV and recurrent state, matched by longest prefix.
    (A flat stand-in for the radix tree: the same match and insert semantics.)"""

    def __init__(self, disable=False):
        self.disable = disable
        self.entries = {}

    def match_prefix(self, tokens):
        """(kv list, recurrent states list) of the longest cached prefix, leaving at least the
        last token to compute (as SGLang does)."""
        best, hit = 0, None
        if self.disable:
            return [], []
        for key, value in self.entries.items():
            n = 0
            for a, b in zip(key, tokens[:-1]):
                if a != b:
                    break
                n += 1
            if n > best:
                best, hit = n, value
        if hit is None:
            return [], []
        return hit[0][:best], hit[1][:best]

    def insert(self, tokens, kv, states):
        if not self.disable:
            self.entries[tuple(tokens)] = (list(kv), list(states))

    def total_size(self):
        return sum(len(key) for key in self.entries)

    def reset(self):
        self.entries = {}

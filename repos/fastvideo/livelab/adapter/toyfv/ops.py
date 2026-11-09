"""Toy engine code the conformance edits change: a pipeline of stages, as FastVideo's."""
SCALE = 2


def layout(batch):
    return batch["frames"]


def geometry(batch):
    return batch["frames"] + 2


def load_schedule():
    """Startup only: the pipeline keeps the steps it returns (as H3's DMD rungs)."""
    return [3, 2, 1]


class Transformer:
    def __init__(self):
        self._pre = []

    def register_forward_pre_hook(self, fn):
        pre = self._pre

        class Handle:
            def remove(self):
                if fn in pre:
                    pre.remove(fn)
        pre.append(fn)
        return Handle()

    def __call__(self, x, step):
        for hook in list(self._pre):
            hook(self, (x, step))
        return [v * 3 + step for v in x]


def verify_input(batch):
    checks = [lambda value: isinstance(value, int)]
    return all(c(batch["seed"]) for c in checks)


class Pipeline:
    def __init__(self):
        from adapter.toyfv import stages
        self.steps = load_schedule()
        self.modules = {"transformer": Transformer()}
        self.video = stages.VideoDecodingStage()
        self.audio = stages.AudioDecodingStage()

    def forward(self, batch):
        assert verify_input(batch)
        x = [batch["seed"] + i for i in range(geometry(batch))]
        for s in self.steps:
            x = self.modules["transformer"](x, s)
        batch["latents"] = x
        return {"video": self.video.forward(batch), "audio": self.audio.forward(batch)}

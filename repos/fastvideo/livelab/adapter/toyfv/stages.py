"""Toy decode stages (a second module, as FastVideo's stages live apart from the pipeline)."""
from adapter.toyfv.ops import SCALE, layout


class VideoDecodingStage:
    def forward(self, batch):
        n = layout(batch)
        return [(v * SCALE + 1) % 251 for v in batch["latents"][:n]]


class AudioDecodingStage:
    def forward(self, batch):
        return [v % 7 for v in batch["latents"]]

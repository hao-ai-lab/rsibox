"""The conformance edits on real MiniMax H3 code (core/livelab/conformance, `patches()`).

Each edit is a string replacement on the base text of one H3 module, so a FastVideo change
that moves these lines makes `h3_patches` fail loudly (AssertionError) instead of building an
edit that tests nothing. What each one touches, and why it is the right probe:

- visible / chain / multi: the video decode stage's uint8 quantization (runs once per video,
  on the output rank) and the audio decode stage: both change the output hashes exactly.
- noop: the same line split in two statements: new code runs, output bit-exact.
- multi: three functions in two modules, all run on every rank (the video decode stage, and
  `_video_geometry` and `_audio_latent_channels` in the latent preparation module), one of them
  visible. (`_layout` runs on the output rank only: an edit of it is honestly not proven on 2
  ranks.)
- unreached: `apply_nvfp4_decoder_checkpoint` runs once at load and sets the VAE's decode
  autocast dtype (bf16) that every decode then reads, so a live install never runs it, while
  a fresh start decodes in fp16 and changes the video. (Changing the DMD rungs instead fails in
  the fresh process: H3's precomputed AdaLN table covers only the trained ladder.)
- refused: a syntax error, a lambda that starts closing over its enclosing function's local,
  a module constant, and a comment-only edit.
"""
DEC = "fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_decoding"
LAT = "fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_latent_preparation"
INP = "fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_input_preparation"
NVD = "fastvideo.models.vaes.minimax_h3_nvfp4_decoder"

AUD_CH = ('def _audio_latent_channels(fastvideo_args: FastVideoArgs) -> int:\n'
          '    return h3_latent_channels(\n'
          '        getattr(fastvideo_args.pipeline_config, "audio_vae_config", None),\n'
          '        "audio_vae_config",\n'
          '    )\n')
from rsibox import harness

QUANT = "output = output.mul_(255).clamp_(0, 255).to(torch.uint8)"
AUDIO = "decoded = self.audio_vae.decode(latents).sample.float()"
_edit = harness.edit_once


def _indent(src, needle):
    line = next(x for x in src.splitlines() if needle in x)
    return line[:len(line) - len(line.lstrip())]


def h3_patches(base):
    """base: {module: text} for DEC, LAT, INP, NVD. Returns the harness's patches dict."""
    dec, lat, inp, nvd = base[DEC], base[LAT], base[INP], base[NVD]
    bf16 = "vae.decode_autocast_dtype = torch.bfloat16"
    assert nvd.count(bf16) == 2, "edit point not found twice: " + bf16
    ind = _indent(dec, QUANT)
    v1 = _edit(dec, QUANT, QUANT.replace(".mul_(255)", ".mul_(255).add_(1)"))
    v2 = _edit(v1, ".add_(1)", ".add_(2)")
    v3 = _edit(v2, AUDIO, AUDIO + " * 0.5")

    def p(**files):
        return {"files": files}
    return {
        "noop": p(**{DEC: _edit(dec, QUANT, "output = output.mul_(255)\n" + ind
                                + "output = output.clamp_(0, 255).to(torch.uint8)")}),
        "visible": p(**{DEC: v1}),
        "chain": [p(**{DEC: v1}), p(**{DEC: v2}), p(**{DEC: v3})],
        "multi": p(**{
            DEC: v1,
            LAT: _edit(_edit(lat, "    _, channels, num_frames, height, width = shape\n",
                             "    channels, num_frames, height, width = shape[1:]\n"),
                       AUD_CH, AUD_CH.replace("    return h3_latent_channels(", "    channels = h3_latent_channels(")
                       + "    return channels\n")}),
        "unreached": p(**{NVD: nvd.replace(bf16, "vae.decode_autocast_dtype = torch.float16")}),
        "refused": {
            "syntax": p(**{DEC: _edit(dec, ".clamp_(0, 255).to(", ".clamp_(0, 255.to(")}),
            "closure": p(**{INP: _edit(inp, 'batch.prompt, lambda value: isinstance(value, str))',
                                       'batch.prompt, lambda value: isinstance(value, str) '
                                       'and result is not None)')}),
            "module_state": p(**{LAT: _edit(lat, 'MINIMAX_H3_LAYOUT_KEY = "minimax_h3_layout"',
                                            'MINIMAX_H3_LAYOUT_KEY = "minimax_h3_layout_v2"')}),
            "empty": p(**{DEC: _edit(dec, "def _layout(batch: ForwardBatch) -> MiniMaxH3PackedLayout:",
                                     "def _layout(batch: ForwardBatch) -> MiniMaxH3PackedLayout:"
                                     "  # faster")}),
        },
    }

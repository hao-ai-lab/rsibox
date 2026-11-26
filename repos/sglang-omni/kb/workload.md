# Workload F(x)
4-stage pipeline (`thinker` RO `Qwen3-VL-4B-Instruct` on `cuda:0` -> mutable `talker` -> `dit` -> RO `vocoder` on `cuda:1`), outputting deterministic `{"thinker_ids", "dit", "waveform", "digest"}`.
- **Shapes (`mode="real"`):** `img64_tok8_dit2` (`64x64`, `max_new_tokens=8`, `dit_steps=2`) and `img128_tok12_dit2` (`128x128`, `max_new_tokens=12`, `dit_steps=2`).
- **Gate:** `witness > 0` on edited stage, frozen stages abstain with `0 B` VRAM delta, `S5` `S0` reset + `uninstall()` + `fresh_check()` match bit-exact (`max_abs_diff == 0.0`).

# Workload F(x)
4-stage pipeline (`thinker` RO `vllm.LLM` `Qwen3-VL-4B-Instruct` on `cuda:0` -> mutable `talker` -> `dit` -> RO `vocoder` on `cuda:0`), outputting `{"thinker_ids", "dit", "waveform", "digest"}`; shapes `img64_tok8_dit2` & `img128_tok12_dit2`; gate `witness > 0`, `0 B` frozen VRAM delta, `S5` `S0` reset + `fresh_check` `max_abs_diff == 0.0`.

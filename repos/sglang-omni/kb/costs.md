# Measured Costs
- **2× B200 (`GPU 2,3`, `juc049/sglang-omni-qwen3:deps`, `Qwen3-VL-4B-Instruct` + CUDA Talker/DiT/Vocoder):** cold boot+check `25.1992 s` (`23.6205 s` + `1.5787 s`), warm `0.0813 s`, `install` `0.0874 s`, `uninstall` `0.1570 s`.
- **2-shape parity:** `img64_tok8_dit2` (`0.0049 s` edited) and `img128_tok12_dit2` (`0.1044 s` edited) both match `fresh_check` bit-exact (`max_abs_diff = 0.0`).
- **Frozen VRAM invariance:** `cuda:0` `8,910,254,592` B, `cuda:1` `33,554,944` B (`0 B` delta across swap, `S0` reset, uninstall).
- **CPU (`ToySglangOmniLab`):** `11/11` `c0–c10` (`-W error::ResourceWarning`); `c9_soak` install `0.0089 s`, uninstall `0.0073 s`.

# Measured Costs
- **1× B200 (`GPU 2`, `vllm 0.33.0` + `torch 2.14.0+cu130`, `Qwen3-VL-4B-Instruct`, standalone `vllm_omni` pkg absent):** cold boot+check `35.5983 s` (`35.4228 s` + `0.1755 s`), warm `0.0082 s`, `install` `0.0276 s`, `uninstall` `0.5513 s`; `11/11` real conformance (`c9` `0.0224 s` / `0.3007 s`, `c8` `30.58–31.63 s`); `6/6` unit+GPU tests (`0` skips, `77.701 s`).
- **2-shape parity & VRAM invariance (`cuda:0`):** `img64_tok8_dit2` (`0.0047 s` edited, `28,284,662,272` B) and `img128_tok12_dit2` (`0.0053 s` edited, `28,286,185,984` B) match `fresh_check` (`max_abs_diff = 0.0`, `0 B` delta across warm/swap/S0/uninstall).
- **CPU (`ToyVllmOmniLab`):** `11/11` `c0–c10` (`-W error::ResourceWarning`; `c9` `0.0083 s` / `0.0071 s`).

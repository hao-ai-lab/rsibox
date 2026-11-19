#!/bin/bash
# Stage a Python virtual environment and an engine source tree on node-local storage, so a
# restart imports from local disk instead of a network file system (environment restart:
# 29-38 s to the first step instead of 58-110 s in our runs; the copy itself 161-247 s once).
#
#   stage_local.sh VENV TREE DEST [JOBS]   ->  DEST/venv, DEST/<tree name>, DEST/cache, DEST/env.sh
#   stage_local.sh --tree-only TREE DEST [JOBS]   ->  DEST/<tree name> (container environment)
#
# DEST/cache holds the compile caches a restart reads: inductor (FX graphs), triton, and the
# CUDA driver's JIT cache (PTX compiled for this GPU; cuDNN fused attention and TE norm kernels
# land there: first call 1.3 s cold vs 0.1-0.25 s warm in our single-layer probe). Set
# CACHE_FROM to an existing cache root with inductor/ and triton/ inside to seed them.
#
# The copy keeps the venv's layout (bin/ symlinks, pyvenv.cfg, every site-packages entry and
# its .pth files; console-script shebangs repointed), so DEST/venv/bin/python finds DEST's site-packages and the base interpreter
# stays where it was. The tree is copied without .git. DEST/env.sh puts both first: source it
# after the job's own environment. Every entry is copied with `cp -a`, JOBS at a time
# (default 32): a network file system serves many small files faster in parallel.
# Prints the seconds of each copy. Run it once per node; it does not check freshness.
set -euo pipefail
VENV=$1; TREE=$2; DEST=$3; JOBS=${4:-32}
now() { date +%s.%N; }
stage_tree() {
  local target="$DEST/$(basename "$TREE")" start
  mkdir -p "$target"
  start=$(now)
  find "$TREE" -mindepth 1 -maxdepth 1 ! -name .git -print0 |
    xargs -0 -r -P "$JOBS" -n 1 cp -at "$target" --
  echo "tree $(awk -v a=$start -v b=$(now) 'BEGIN{printf "%.1f", b-a}') s"
}
if [ "$VENV" = --tree-only ]; then
  stage_tree
  exit
fi
PY=$(ls -d "$VENV"/lib/python3.* | head -1); PYV=$(basename "$PY")
SP=$PY/site-packages; LSP=$DEST/venv/lib/$PYV/site-packages
mkdir -p "$LSP"
t=$(now)
cp -a "$VENV/bin" "$VENV/pyvenv.cfg" "$DEST/venv/"
find "$SP" -mindepth 1 -maxdepth 1 -print0 | xargs -0 -r -P "$JOBS" -n 1 cp -at "$LSP" --
# console scripts (torchrun, ...) name the source venv's python in their shebang: repoint them,
# or their processes import from the network file system again
for f in "$DEST"/venv/bin/*; do
  [ -f "$f" ] && [ ! -L "$f" ] && head -c 2 "$f" | grep -q '#!' && sed -i "1s|^#!$VENV/bin/python|#!$DEST/venv/bin/python|" "$f"
done
echo "venv $(awk -v a=$t -v b=$(now) 'BEGIN{printf "%.1f", b-a}') s"
stage_tree
mkdir -p "$DEST/cache/inductor" "$DEST/cache/triton" "$DEST/cache/cuda"
for c in inductor triton; do
  if [ -n "${CACHE_FROM:-}" ] && [ -d "$CACHE_FROM/$c" ]; then cp -a "$CACHE_FROM/$c/." "$DEST/cache/$c/"; fi
done
cat > "$DEST/env.sh" <<EOT
export PATH=$DEST/venv/bin:\$PATH
export PYTHONPATH=$DEST/$(basename "$TREE")\${PYTHONPATH:+:\$PYTHONPATH}
export LD_LIBRARY_PATH=$LSP/nvidia/nccl/lib:$LSP/nvidia/cudnn/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}
export TORCHINDUCTOR_CACHE_DIR=$DEST/cache/inductor TRITON_CACHE_DIR=$DEST/cache/triton
export TORCHINDUCTOR_COMPILE_THREADS=\${TORCHINDUCTOR_COMPILE_THREADS:-8}
export CUDA_CACHE_PATH=$DEST/cache/cuda CUDA_CACHE_MAXSIZE=4294967296
EOT
echo "env $DEST/env.sh"

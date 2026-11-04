#!/bin/bash
# Fail if lines added between BASE and HEAD look like environment details or secrets.
# A pattern check, not a guarantee: it only sees the shapes listed below.
# Usage: leak-check.sh <base> <head>
set -euo pipefail
base=$1 head=$2

# Files that must name these patterns to describe the rule.
exclude=(':(exclude)AGENTS.md' ':(exclude).github/scripts/leak-check.sh'
         ':(exclude).github/workflows/pr-rules.yml')

# name|extended regex (case-insensitive)
patterns=(
  'private IPv4 10/8|(^|[^0-9.])10\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}([^0-9.]|$)'
  'private IPv4 172.16/12|(^|[^0-9.])172\.(1[6-9]|2[0-9]|3[01])\.[0-9]{1,3}\.[0-9]{1,3}([^0-9.]|$)'
  'private IPv4 192.168/16|(^|[^0-9.])192\.168\.[0-9]{1,3}\.[0-9]{1,3}([^0-9.]|$)'
  'CGNAT/Tailscale IPv4 100.64/10|(^|[^0-9.])100\.(6[4-9]|[7-9][0-9]|1[01][0-9]|12[0-7])\.[0-9]{1,3}\.[0-9]{1,3}([^0-9.]|$)'
  'home or scratch path|/(raid|workspace|scratch)/|/home/[a-z]|/Users/[A-Za-z]|/mnt/lustre'
  'cluster access recipe|kubeconfig|ProxyCommand|tailscale|\.ts\.net|ts\.sock|kubectl .*exec'
  'private key|BEGIN [A-Z ]*PRIVATE KEY'
  'token|(ghp|gho|ghs|github_pat)_[A-Za-z0-9_]{20,}|hf_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}'
)

# Added lines as "file:line-in-new-file:text".
added=$(git diff -U0 --no-color "$base...$head" -- . "${exclude[@]}" | awk '
  /^\+\+\+ b\// { file = substr($0, 7); next }
  /^@@/ { split($3, a, ","); line = substr(a[1], 2) + 0; next }
  /^\+/ { print file ":" line ": " substr($0, 2); line++ }')

found=0
for entry in "${patterns[@]}"; do
  name=${entry%%|*} regex=${entry#*|}
  hits=$(printf '%s\n' "$added" | grep -Ei -- "$regex" || true)
  if [ -n "$hits" ]; then
    found=1
    echo "LEAK? $name:"
    printf '%s\n' "$hits" | cut -c1-160 | sed 's/^/  /'
  fi
done

lines=$(printf '%s' "$added" | grep -c '' || true)
if [ $found = 1 ]; then
  echo "Move these to ~/fastrsi-lab/<engine>/ (or .lab/) and describe them generically. See AGENTS.md."
  exit 1
fi
echo "OK: no leak pattern in $lines added lines. This is a pattern check, not a guarantee."

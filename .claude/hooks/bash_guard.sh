#!/usr/bin/env bash
# PreToolUse guard for Bash commands.
#  1. SPEC §3.2 / Dataset.md §3.1: the Kaggle test.7z (300k images, 290k dummies, no labels) is never
#     downloaded, so a whole-competition download (no -f) is blocked too.
#  2. SPEC §0 C1: shell writes (redirects, tee, sed -i, heredocs) must not smuggle banned constructs
#     past c1_guard.sh, which only sees the Write/Edit tools. Read-only greps are unaffected.
input=$(cat)
cmd=$(jq -r '.tool_input.command // empty' <<<"$input")
[ -z "$cmd" ] && exit 0

if grep -qE 'kaggle[[:space:]]+competitions[[:space:]]+download' <<<"$cmd"; then
  if grep -qE 'test\.7z' <<<"$cmd" || ! grep -qE '(^|[[:space:]])(-f|--file)[[:space:]=]' <<<"$cmd"; then
    {
      echo "Blocked by SPEC §3.2: only train.7z and trainLabels.csv are fetched from Kaggle, one file at a time:"
      echo "  kaggle competitions download cifar-10 -f train.7z -p data/cifar10"
      echo "  kaggle competitions download cifar-10 -f trainLabels.csv -p data/cifar10"
      echo "The labelled probe pool comes from torchvision.datasets.CIFAR10(train=False)."
    } >&2
    exit 2
  fi
fi

# Redirects to /dev/null and stderr-to-stdout merges are not writes.
writes=$(sed -E 's/[0-9&]*>>?[[:space:]]*\/dev\/null//g; s/[0-9]*>&[0-9]//g' <<<"$cmd")
if grep -qE '>|\btee\b|sed[[:space:]]+(-[a-zA-Z]*i|--in-place)|<<' <<<"$writes" \
   && grep -qiE 'mahalanobis|EllipticEnvelope|MinCovDet|EmpiricalCovariance|LedoitWolf|ShrunkCovariance|GraphicalLasso|zca' <<<"$cmd"; then
  {
    echo "Blocked by SPEC C1: this command writes text containing a banned Mahalanobis/whitening construct."
    echo "Searching for banned names is fine; writing them into files is not (except tests/*c1* via the Write tool)."
  } >&2
  exit 2
fi
exit 0

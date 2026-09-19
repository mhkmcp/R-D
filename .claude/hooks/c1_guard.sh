#!/usr/bin/env bash
# PreToolUse guard for SPEC.md §0 C1: block writes that introduce Mahalanobis distance,
# covariance estimators feeding one, or ZCA/covariance whitening into code or configs.
input=$(cat)
path=$(jq -r '.tool_input.file_path // .tool_input.notebook_path // empty' <<<"$input")

case "$path" in
  */.claude/*) exit 0 ;;  # the guards and skills must spell out what they ban
  */tests/*c1*) exit 0 ;;  # the C1 grep-guard test must spell out the patterns it searches for
esac
case "$path" in
  *.py|*.ipynb|*.yaml|*.yml|*.toml|*.cfg|*.ini|*.json|*.sh|*/Makefile) ;;
  *) exit 0 ;;  # docs may name the banned constructs in order to ban them
esac

text=$(jq -r '[.tool_input.content, .tool_input.new_string, .tool_input.new_source,
               (.tool_input.edits // [] | .[].new_string)] | map(select(. != null)) | join("\n")' <<<"$input")

# Names, then structural equivalents: whitened PCA and an explicit inverse covariance.
hits=$(grep -niE 'mahalanobis|EllipticEnvelope|MinCovDet|EmpiricalCovariance|LedoitWolf|ShrunkCovariance|GraphicalLasso|zca|whiten[[:space:]]*=[[:space:]]*True|(inv|pinv)[[:space:]]*\([[:space:]]*(np|numpy|torch)\.cov' <<<"$text")
if [ -n "$hits" ]; then
  {
    echo "Blocked by SPEC C1 (Mahalanobis is excluded from the entire project) in $path:"
    echo "$hits"
    echo "Use the permitted substitutes instead: median/IQR standardisation, RBF-kernel SVDD, KDE, LOF, Isolation Forest, reconstruction error."
  } >&2
  exit 2
fi
exit 0

#!/usr/bin/env bash
# PostToolUse: lint Python files under src/ and tests/ right after they are written, so problems are
# fixed in the same turn rather than found at `uv run ruff check`. Exit 2 feeds the output back to
# Claude; the edit itself has already happened.
input=$(cat)
path=$(jq -r '.tool_input.file_path // empty' <<<"$input")

case "$path" in
  "$CLAUDE_PROJECT_DIR"/src/*.py|"$CLAUDE_PROJECT_DIR"/tests/*.py) ;;
  *) exit 0 ;;
esac
[ -f "$path" ] || exit 0

ruff="$CLAUDE_PROJECT_DIR/.venv/bin/ruff"
[ -x "$ruff" ] || exit 0  # env not synced yet; `uv sync` installs it

if ! out=$("$ruff" check --quiet --config "$CLAUDE_PROJECT_DIR/pyproject.toml" "$path" 2>&1); then
  echo "ruff check failed for $path:" >&2
  echo "$out" >&2
  exit 2
fi
exit 0

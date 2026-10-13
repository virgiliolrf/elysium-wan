#!/bin/sh
# PostToolUse hook entry for Bash (see approval-record.mjs). Fast path: only a command that ran `approve.mjs confirm`
# reaches node. Skipping can only mean "no grant recorded", never a grant without the human.
p=$(cat)
case "$p" in
  *"approve.mjs confirm"*) printf '%s' "$p" | node "$(dirname "$0")/approval-record.mjs"; exit $? ;;
esac
exit 0

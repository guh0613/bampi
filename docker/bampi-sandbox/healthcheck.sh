#!/bin/sh
set -eu

test -d /workspace/inbox
test -d /workspace/outbox
test "${HOME:-}" = /home/bampi

agently_cli="$(command -v agently-cli)"
test -n "$agently_cli"
test -x "$agently_cli"

probe="$(mktemp /tmp/bampi-healthcheck.XXXXXX)"
trap 'rm -f "$probe"' EXIT HUP INT TERM
printf '#!/bin/sh\nexit 0\n' >"$probe"
chmod 0700 "$probe"
"$probe"

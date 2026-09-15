#!/bin/sh
# Starts the web app in the background and makes its link public; run by
# postStartCommand on every boot.
#
# The dev-container CLI kills whatever is left in the command's process group the
# moment this script exits, so a plain `nohup ... &` dies before it has even opened
# its log. setsid moves the server into a session of its own, and waiting for it to
# answer means the script never exits while the server is still starting up.
cd "$(dirname "$0")/.." || exit 1
LOG=/tmp/scaffold-app.log
up() { curl -fs -o /dev/null http://127.0.0.1:8000/; }

if up; then
  echo "app already running on :8000"
else
  setsid nohup python -u webapp/server.py 8000 > "$LOG" 2>&1 < /dev/null &
  # A cold boot takes ~30 s to import the scientific stack before the first answer.
  i=0
  until up; do
    i=$((i + 1))
    if [ "$i" -gt 120 ]; then
      echo "app did not answer on :8000 within 120 s; last lines of $LOG:"
      tail -n 20 "$LOG"
      exit 1
    fi
    sleep 1
  done
  echo "app running on :8000 (log: $LOG)"
fi

# Stopping a codespace resets its ports to private, which would turn a shared link
# into a GitHub sign-in page. The codespace's own token is allowed to set it back.
[ -n "$CODESPACE_NAME" ] || . /workspaces/.codespaces/shared/.env 2>/dev/null
if [ -n "$CODESPACE_NAME" ] && [ -n "$GITHUB_TOKEN" ] &&
   GITHUB_TOKEN="$GITHUB_TOKEN" gh codespace ports visibility 8000:public -c "$CODESPACE_NAME"; then
  echo "port 8000 is public: https://$CODESPACE_NAME-8000.${GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN:-app.github.dev}"
else
  echo "could not make port 8000 public; run: gh codespace ports visibility 8000:public -c <codespace-name>"
fi

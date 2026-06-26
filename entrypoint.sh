cleanup() {
  EXIT_CODE=$?
  echo "=== Container Exit ==="
  echo "Command exited with code: $EXIT_CODE"
  
  if [ "$EXIT_CODE" -ne 0 ]; then
    echo "ERROR: Our code crashed (Python exception, OOM, bad config, etc)!"
    echo "Sleeping for 5 minutes to ensure Vast.ai syncs these error logs before self-destructing..."
    sleep 300
  else
    echo "SUCCESS: Task completed normally."
  fi

  if [ "$NO_DESTROY" = "1" ]; then
    echo "NO_DESTROY=1 is set. Skipping instance destruction."
    echo "Instance $CONTAINER_ID will remain active. Remember to destroy it manually!"
    # Sleep indefinitely to keep the container alive
    sleep infinity
  else
    echo "Destroying instance $CONTAINER_ID to stop billing..."
    vastai destroy instance "$CONTAINER_ID"
  fi
}
trap cleanup EXIT

# ldconfig

echo 'PIP freeze (subset):'
pip freeze | grep nvidia
pip freeze | grep jax

#echo GCP instance:
#echo "Name:     $(curl -s -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/name || echo NA)"
#echo "Hostname: $(curl -s -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/hostname || echo NA)"
#echo "ID:       $(curl -s -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/id || echo NA)"
#echo "Zone:     $(curl -s -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/zone || echo NA)"
#echo

# Initialize Tailscale if Auth Key is provided
if [ ! -z "$TAILSCALE_AUTHKEY" ]; then
  # Log the key id (the segment before the secret) — NOT the secret — so the
  # instance log shows WHICH key was used and you can cross-check it against the
  # Tailscale admin console. This is the segment Tailscale itself names if it
  # rejects the key ("invalid key: API key <keyid> not valid").
  TS_KEYID=$(echo "$TAILSCALE_AUTHKEY" | cut -d- -f3)
  echo "TAILSCALE_AUTHKEY detected (keyid: ${TS_KEYID}). Starting Tailscale daemon in userspace mode..."
  # Start the daemon in userspace networking mode (bypasses Docker capability requirements)
  tailscaled --tun=userspace-networking --socks5-server=localhost:1055 --outbound-http-proxy-listen=localhost:1055 &

  # Wait for daemon to initialize
  sleep 3

  echo "Authenticating Tailscale..."
  # Check the exit code — do NOT claim "UP" when the key was rejected. A silent
  # failure here is why the 2026-06 instances never joined the tailnet despite a
  # key being present: the key was passed correctly but rejected server-side
  # (expired / revoked / single-use already consumed), and the old unconditional
  # "Tailscale is UP!" line hid that.
  if tailscale up --authkey="${TAILSCALE_AUTHKEY}" --ssh --hostname="vast-$(hostname)" --timeout=30s; then
    echo "Tailscale is UP! You can SSH via: ssh root@vast-$(hostname)"
  else
    echo "ERROR: 'tailscale up' FAILED for keyid ${TS_KEYID} — the auth key was REJECTED"
    echo "       (expired, revoked, or a single-use key already consumed)."
    echo "       Fix: mint a fresh REUSABLE + EPHEMERAL + PRE-APPROVED key at"
    echo "       https://login.tailscale.com/admin/settings/keys and update"
    echo "       TAILSCALE_AUTHKEY in .env. Training continues without the tailnet;"
    echo "       Vast proxy SSH (vast.py ssh-url <id>) still works in the meantime."
  fi
fi

echo GPUs:
nvidia-smi --query-gpu=gpu_name,memory.total,driver_version --format=csv || true
echo

xvfb-run -a -s '-screen 0 1024x768x24 -ac +extension GLX +render -noreset' "$@"
# xvfb-run "$@"
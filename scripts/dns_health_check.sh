#!/usr/bin/env bash
# DNS staleness check for noraxdev.org (audit item 4).
# Exits 0 when DNS resolves to expected Cloudflare ranges; exits 1 and
# writes state/dns_health.alert otherwise so the service monitor surfaces it.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE_DIR="$ROOT/state"
ALERT_FILE="$STATE_DIR/dns_health.alert"
DOMAIN="${NORAX_DNS_DOMAIN:-noraxdev.org}"

mkdir -p "$STATE_DIR"

# Retry transient resolution failures: resolver blips (network restart,
# systemd-resolved restart) should not mark the unit failed. Only a
# persistent failure across retries is real staleness.
ADDRS=()
for attempt in 1 2 3 4; do
    mapfile -t ADDRS < <(getent hosts "$DOMAIN" | awk '{print $1}' | sort -u)
    if [[ ${#ADDRS[@]} -gt 0 ]]; then
        break
    fi
    echo "dns_health: attempt $attempt: no addresses for $DOMAIN, retrying..." >&2
    sleep $((attempt * 5))
done

if [[ ${#ADDRS[@]} -eq 0 ]]; then
    echo "dns_health FAIL domain=$DOMAIN reason=no_resolution_after_retries" > "$ALERT_FILE"
    echo "dns_health: no addresses resolved for $DOMAIN after 4 attempts" >&2
    exit 1
fi

# Expected: Cloudflare fronted. IPv4 104.21.x / 172.67.x, IPv6 2606:4700::/xx
bad=()
for a in "${ADDRS[@]}"; do
    case "$a" in
        104.21.*|172.67.*|2606:4700:*) : ;;  # ok
        *) bad+=("$a") ;;
    esac
done

if [[ ${#bad[@]} -gt 0 ]]; then
    printf 'dns_health FAIL domain=%s unexpected=%s all=%s\n' \
        "$DOMAIN" "$(printf '%s,' "${bad[@]}")" "$(printf '%s,' "${ADDRS[@]}")" > "$ALERT_FILE"
    echo "dns_health: unexpected addresses for $DOMAIN: ${bad[*]}" >&2
    exit 1
fi

rm -f "$ALERT_FILE"
echo "dns_health OK domain=$DOMAIN addrs=$(printf '%s,' "${ADDRS[@]}")"

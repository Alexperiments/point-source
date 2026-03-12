#!/bin/sh
set -eu

CERT_DIR="/etc/letsencrypt/live/www.point-source.org"
FULLCHAIN="$CERT_DIR/fullchain.pem"
PRIVKEY="$CERT_DIR/privkey.pem"
RENEWAL_FLAG_FILE="${RENEWAL_FLAG_FILE:-/var/www/certbot/.certbot-renewed}"
WATCH_INTERVAL_SECONDS="${NGINX_CERT_WATCH_INTERVAL_SECONDS:-60}"

watch_for_certificate_updates() {
  while :; do
    if [ -f "$RENEWAL_FLAG_FILE" ]; then
      nginx -s reload && rm -f "$RENEWAL_FLAG_FILE"
    fi
    sleep "$WATCH_INTERVAL_SECONDS"
  done
}

if [ ! -s "$FULLCHAIN" ] || [ ! -s "$PRIVKEY" ]; then
  mkdir -p "$CERT_DIR"
  openssl req -x509 -nodes -newkey rsa:2048 -days 1 \
    -keyout "$PRIVKEY" \
    -out "$FULLCHAIN" \
    -subj "/CN=www.point-source.org"
fi

watch_for_certificate_updates &

exec nginx -g "daemon off;"

#!/bin/sh
set -eu

CERT_DIR="/etc/letsencrypt/live/www.point-source.org"
FULLCHAIN="$CERT_DIR/fullchain.pem"
PRIVKEY="$CERT_DIR/privkey.pem"

if [ ! -s "$FULLCHAIN" ] || [ ! -s "$PRIVKEY" ]; then
  mkdir -p "$CERT_DIR"
  openssl req -x509 -nodes -newkey rsa:2048 -days 1 \
    -keyout "$PRIVKEY" \
    -out "$FULLCHAIN" \
    -subj "/CN=www.point-source.org"
fi

exec nginx -g "daemon off;"

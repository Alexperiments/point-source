#!/bin/sh
set -eu

RENEWAL_FLAG_FILE="${RENEWAL_FLAG_FILE:-/var/www/certbot/.certbot-renewed}"
RENEW_INTERVAL_SECONDS="${CERTBOT_RENEW_INTERVAL_SECONDS:-43200}"

while :; do
  certbot renew \
    --webroot \
    -w /var/www/certbot \
    --quiet \
    --deploy-hook "sh -c 'touch \"${RENEWAL_FLAG_FILE}\"'"
  sleep "${RENEW_INTERVAL_SECONDS}"
done

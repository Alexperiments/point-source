# TLS Setup

`nginx` is configured for HTTPS on `www.point-source.org` and shares ACME files with the `certbot` service.

## 1. DNS

Point `www.point-source.org` to the server's public IP.
Point `point-source.org` to the same IP if you want the apex domain to redirect to `www`.

## 2. Production env

In `backend/.env.production`, set:

```env
ALLOWED_ORIGINS=https://www.point-source.org
```

The frontend production build uses same-origin API calls via `frontend/.env.production`, so it should stay behind nginx on the same domain.

## 3. Start the stack

```bash
docker compose --env-file backend/.env.production -f infra/docker-compose.yml up -d nginx point-source postgres redis litellm certbot
```

On first boot, `nginx` uses a temporary self-signed certificate so it can start before Let's Encrypt issues the real one.

## 4. Issue the first certificate

```bash
docker compose --env-file backend/.env.production -f infra/docker-compose.yml run --rm certbot \
  certonly \
  --webroot \
  -w /var/www/certbot \
  -d www.point-source.org \
  --email YOUR_EMAIL_HERE \
  --agree-tos \
  --no-eff-email
```

## 5. Reload nginx

```bash
docker compose -f infra/docker-compose.yml exec nginx nginx -s reload
```

## 6. Renewal

The `certbot` service runs `certbot renew` every 12 hours. When a certificate is renewed, it touches a shared flag file and the `nginx` container automatically reloads within about 60 seconds.

You do not need a host-level cron job for certificate renewal with this setup.

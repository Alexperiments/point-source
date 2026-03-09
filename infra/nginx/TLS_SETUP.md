# TLS Setup

`nginx` is configured for HTTPS on `www.point-sources.com` and shares ACME files with the `certbot` service.

## 1. DNS

Point `www.point-sources.com` to the server's public IP.

## 2. Start the stack

```bash
docker compose --env-file backend/.env.production -f infra/docker-compose.yml up -d nginx point-source postgres redis litellm certbot
```

On first boot, `nginx` uses a temporary self-signed certificate so it can start before Let's Encrypt issues the real one.

## 3. Issue the first certificate

```bash
docker compose --env-file backend/.env.production -f infra/docker-compose.yml run --rm certbot \
  certonly \
  --webroot \
  -w /var/www/certbot \
  -d www.point-sources.com \
  --email YOUR_EMAIL_HERE \
  --agree-tos \
  --no-eff-email
```

## 4. Reload nginx

```bash
docker compose -f infra/docker-compose.yml exec nginx nginx -s reload
```

## 5. Renewal

The `certbot` service runs `certbot renew` every 12 hours. After a renewal, reload `nginx`:

```bash
docker compose -f infra/docker-compose.yml exec nginx nginx -s reload
```

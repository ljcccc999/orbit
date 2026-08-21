# Orbit Hub

Orbit Hub is an optional account, upload, and administrator-review service. It receives finished models and GPU training packages that Orbit generates locally. It never trains, loads, or executes uploaded files automatically. Existing Orbit installations continue to work locally when no Hub is configured.

## Minimum practical server

- 1–2 vCPU and 1–2 GB RAM are enough for accounts, metadata, and a few concurrent uploads.
- Disk and transfer are the actual limit. A 300M FP32 checkpoint is roughly 1.2 GB; 1B is roughly 4 GB; 38B can exceed 150 GB before packaging. A small server cannot economically retain many large models.
- Start with 40–80 GB SSD and a per-model limit of 5–10 GB. Models over the limit remain local. Later, object storage can replace the local model volume without moving training to the server.
- A domain pointed at the server is required. Caddy obtains and renews HTTPS automatically.

## Deploy

```bash
cp .env.example .env
# Edit ORBIT_DOMAIN and optional storage limits.
docker compose up -d --build
docker compose exec hub cat /data/initial-admin-token
```

Open `https://your-domain/`, paste the one-time token, and choose the administrator email and a password of at least 12 characters. The token is deleted immediately after the first administrator is created. No default or hard-coded administrator password exists.

Back up the named `orbit-hub-data` volume. It contains the SQLite database, pending uploads, approved files, account hashes, and review records. Backups and the server itself should use encrypted storage where available.

## Security boundaries

- Passwords use Argon2id; plaintext passwords are never stored.
- Sessions are random opaque values; browser sessions use Secure, HttpOnly, SameSite=Strict cookies and CSRF tokens.
- Desktop clients may use an opaque bearer session stored in their local user-only Orbit configuration.
- Uploads use fixed 8 MiB chunks, server quotas, extension allowlisting, random server-side names, final size/SHA-256 verification, and an administrator review gate. The same verified `.zip` path is used for human-authored and AI-assisted GPU training packages.
- Uploaded files are not parsed, extracted, imported, executed, or loaded by Hub.
- Approval controls visibility; it does not prove a model is safe, lawful, accurate, or free of malicious behavior.
- Put SSH behind key authentication and a firewall, keep Docker/the OS updated, and do not expose port 8080 directly.

Relevant implementation guidance: [OWASP Password Storage](https://cheatsheetseries.owasp.org/cheatsheets/Password_Storage_Cheat_Sheet.html), [Session Management](https://cheatsheetseries.owasp.org/cheatsheets/Session_Management_Cheat_Sheet.html), [CSRF Prevention](https://cheatsheetseries.owasp.org/cheatsheets/Cross-Site_Request_Forgery_Prevention_Cheat_Sheet.html), and [File Upload](https://cheatsheetseries.owasp.org/cheatsheets/File_Upload_Cheat_Sheet.html).

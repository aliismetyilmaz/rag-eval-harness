# REST API (v2)

_Last updated: 2026-04-02_

The v2 API is the current version. Base URL: `https://api.relaykit.example/v2`.
API access requires the Growth or Enterprise plan.

## Authentication

Requests authenticate with a bearer token in the `Authorization` header. Admins create
tokens under Settings → API. Tokens expire after 90 days and can be revoked at any time.

## Rate limits

- Growth: 300 requests per minute per account.
- Enterprise: 1,000 requests per minute per account.

Requests over the limit receive HTTP 429 with a `Retry-After` header giving the number of
seconds to wait.

## Pagination

List endpoints use cursor-based pagination. Pass the `next_cursor` value from one
response as the `cursor` parameter of the next request. The maximum page size is 100.

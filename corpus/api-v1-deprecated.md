# REST API (v1) — Deprecated

_Last updated: 2025-09-01_

> **Deprecated.** The v1 API is no longer maintained. Use the v2 API instead.
> v1 was sunset on 2026-03-31; since then every v1 request returns HTTP 410 Gone.

Base URL: `https://api.relaykit.example/v1`.

## Authentication

v1 authenticated with an API key passed in the `api_key` query-string parameter.

## Rate limits

v1 allowed 60 requests per minute per account on every plan.

## Pagination

v1 used page-number pagination (`page` and `per_page`), with a maximum of 50 results
per page.

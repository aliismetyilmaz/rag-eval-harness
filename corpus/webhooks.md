# Webhooks

_Last updated: 2026-02-18_

Webhooks send an HTTP POST to your endpoint when something happens in Relaykit.
Each account can register up to 20 webhook endpoints.

## Events

- `ticket.created`
- `ticket.updated`
- `ticket.solved`
- `customer.created`

## Verifying requests

Every request is signed with HMAC-SHA256 using the endpoint's signing secret. The
signature is in the `X-Relaykit-Signature` header.

## Delivery and retries

Your endpoint must respond with a 2xx status within 10 seconds. Failed deliveries are
retried 5 times with exponential backoff over about 24 hours. An endpoint that keeps
failing for 7 consecutive days is disabled automatically, and the account owner is
emailed.

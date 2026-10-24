# Security and Single Sign-On

_Last updated: 2025-11-20_

## Single sign-on (SSO)

SAML 2.0 single sign-on is available on the Enterprise plan only. Supported identity
providers are Okta, Microsoft Entra ID (formerly Azure AD), and Google Workspace.
SCIM user provisioning is also Enterprise-only.

## Two-factor authentication

Every user on every plan can turn on two-factor authentication (2FA) with an
authenticator app. Admins on Growth and Enterprise can require 2FA for all agents.

## Sessions

Agent sessions time out after 12 hours of inactivity by default. Enterprise admins can
set the timeout anywhere from 1 to 72 hours.

## Compliance and encryption

Relaykit holds a SOC 2 Type II report, available to customers under NDA. Data is
encrypted at rest with AES-256 and in transit with TLS 1.2 or higher.

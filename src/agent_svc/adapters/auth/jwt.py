"""JWT bearer auth (HS256 shared secret, RS256 static key, or RS256 via JWKS)."""

from __future__ import annotations

import asyncio
from typing import Any

import jwt
from jwt import PyJWKClient

from agent_svc.config.settings import AuthSettings, Settings
from agent_svc.domain.exceptions import ConfigurationError, UnauthorizedError
from agent_svc.domain.models import Principal


class JwtAuth:
    def __init__(self, auth: AuthSettings, default_tier: str) -> None:
        self._s = auth
        self._default_tier = default_tier
        self._jwks = PyJWKClient(auth.jwks_url, cache_keys=True) if auth.jwks_url else None
        if auth.algorithm == "HS256" and not auth.secret:
            raise ConfigurationError("AUTH enabled with HS256 requires AGENT_AUTH__SECRET")
        if auth.algorithm == "RS256" and not (auth.public_key or auth.jwks_url):
            raise ConfigurationError(
                "RS256 requires AGENT_AUTH__PUBLIC_KEY or AGENT_AUTH__JWKS_URL"
            )

    async def _key(self, token: str) -> Any:
        if self._s.algorithm == "HS256":
            assert self._s.secret is not None
            return self._s.secret.get_secret_value()
        if self._jwks is not None:
            signing = await asyncio.to_thread(self._jwks.get_signing_key_from_jwt, token)
            return signing.key
        return self._s.public_key

    async def authenticate(self, authorization: str | None, *, client_ip: str | None) -> Principal:
        if not authorization or not authorization.lower().startswith("bearer "):
            raise UnauthorizedError("missing bearer token")
        token = authorization[7:].strip()
        try:
            claims = jwt.decode(
                token,
                await self._key(token),
                algorithms=[self._s.algorithm],  # pinned: never trust the token's own `alg`
                issuer=self._s.issuer,
                audience=self._s.audience,
                leeway=self._s.leeway_seconds,
                options={
                    "require": ["exp", "sub"],
                    "verify_iss": self._s.issuer is not None,
                    "verify_aud": self._s.audience is not None,
                },
            )
        except jwt.PyJWTError as exc:
            raise UnauthorizedError("invalid or expired token") from exc  # no detail leakage

        scopes = claims.get("scope", claims.get("scopes", []))
        if isinstance(scopes, str):
            scopes = scopes.split()
        return Principal(
            subject=str(claims["sub"]),
            tier=str(claims.get("tier", self._default_tier)),
            tenant=str(claims.get("tenant", claims.get("tid", "default"))),
            scopes=frozenset(map(str, scopes)),
            ip=client_ip,
        )


def build(settings: Settings) -> JwtAuth:
    return JwtAuth(settings.auth, settings.ratelimit.default_tier)

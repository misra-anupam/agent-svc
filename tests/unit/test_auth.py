import time

import jwt
import pytest
from pydantic import SecretStr

from agent_svc.adapters.auth.jwt import JwtAuth
from agent_svc.adapters.auth.noauth import NoAuth
from agent_svc.config.settings import AuthSettings
from agent_svc.domain.exceptions import ConfigurationError, UnauthorizedError

SECRET = "s3cret-s3cret-s3cret-s3cret-123456"


def auth(**kw: object) -> JwtAuth:
    return JwtAuth(AuthSettings(enabled=True, secret=SecretStr(SECRET), **kw), "free")  # type: ignore[arg-type]


def token(claims: dict[str, object] | None = None, key: str = SECRET, alg: str = "HS256") -> str:
    base = {"sub": "alice", "exp": int(time.time()) + 60}
    return jwt.encode({**base, **(claims or {})}, key, algorithm=alg)


async def test_valid_token_maps_claims_to_principal() -> None:
    t = token({"tier": "pro", "tenant": "acme", "scope": "chat memory"})
    p = await auth().authenticate(f"Bearer {t}", client_ip="1.2.3.4")
    assert (p.subject, p.tier, p.tenant, p.ip, p.anonymous) == (
        "alice",
        "pro",
        "acme",
        "1.2.3.4",
        False,
    )
    assert p.scopes == {"chat", "memory"}


@pytest.mark.parametrize(
    "header",
    [None, "", "Basic abc", "Bearer", "Bearer not.a.jwt"],
)
async def test_missing_or_garbage_credentials_rejected(header: str | None) -> None:
    with pytest.raises(UnauthorizedError):
        await auth().authenticate(header, client_ip=None)


async def test_expired_and_wrong_signature_rejected() -> None:
    with pytest.raises(UnauthorizedError):
        await auth().authenticate(
            "Bearer " + token({"exp": int(time.time()) - 600}), client_ip=None
        )
    with pytest.raises(UnauthorizedError):
        await auth().authenticate("Bearer " + token(key="x" * 40), client_ip=None)


async def test_token_without_exp_or_sub_rejected() -> None:
    no_exp = jwt.encode({"sub": "a"}, SECRET, algorithm="HS256")
    no_sub = jwt.encode({"exp": int(time.time()) + 60}, SECRET, algorithm="HS256")
    for t in (no_exp, no_sub):
        with pytest.raises(UnauthorizedError):
            await auth().authenticate(f"Bearer {t}", client_ip=None)


async def test_algorithm_is_pinned_none_alg_rejected() -> None:
    forged = jwt.encode({"sub": "root", "exp": int(time.time()) + 60}, key="", algorithm="none")
    with pytest.raises(UnauthorizedError):
        await auth().authenticate(f"Bearer {forged}", client_ip=None)
    hs512 = token(alg="HS512")
    with pytest.raises(UnauthorizedError):
        await auth().authenticate(f"Bearer {hs512}", client_ip=None)


async def test_issuer_and_audience_enforced() -> None:
    a = auth(issuer="https://idp", audience="agent-svc")
    ok = token({"iss": "https://idp", "aud": "agent-svc"})
    assert (await a.authenticate(f"Bearer {ok}", client_ip=None)).subject == "alice"
    with pytest.raises(UnauthorizedError):
        await a.authenticate(
            "Bearer " + token({"iss": "https://evil", "aud": "agent-svc"}), client_ip=None
        )
    with pytest.raises(UnauthorizedError):
        await a.authenticate(
            "Bearer " + token({"iss": "https://idp", "aud": "other"}), client_ip=None
        )


def test_misconfiguration_fails_at_startup() -> None:
    with pytest.raises(ConfigurationError):
        JwtAuth(AuthSettings(enabled=True), "free")
    with pytest.raises(ConfigurationError):
        JwtAuth(AuthSettings(enabled=True, algorithm="RS256"), "free")


async def test_noauth_is_anonymous_by_ip() -> None:
    p = await NoAuth("free").authenticate(None, client_ip="9.9.9.9")
    assert p.anonymous and p.ip == "9.9.9.9" and p.rate_key == "ip:9.9.9.9"

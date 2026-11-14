"""SATIE worker sign-in: discovery, PKCE login, token refresh. Fake endpoints only."""

import base64
import hashlib
import json
import threading
import urllib.parse
import urllib.request

import pytest

from memoryscapes_workers.providers import satie_auth
from memoryscapes_workers.providers.mcp_transport import HttpMcpTransport
from memoryscapes_workers.providers.satie_auth import SatieAuthError, SatieTokenAuth

AS = "https://auth.example/auth/v1"
# The shapes SATIE published on 2026-11-14 (Supabase Auth behind www.satie.world/mcp).
METADATA = {
    "https://satie.example/.well-known/oauth-protected-resource/mcp": {
        "resource": "https://satie.example/mcp", "authorization_servers": [AS],
        "scopes_supported": ["email", "profile"]},
    "https://auth.example/.well-known/oauth-authorization-server/auth/v1": {
        "issuer": AS, "authorization_endpoint": AS + "/oauth/authorize", "token_endpoint": AS + "/oauth/token",
        "registration_endpoint": AS + "/oauth/clients/register",
        "scopes_supported": ["openid", "profile", "email", "phone", "offline_access"],
        "code_challenge_methods_supported": ["S256", "plain"]},
}


def fake_get(table):
    def get(url):
        if url not in table:
            raise OSError(f"404 {url}")
        return table[url]
    return get


class FakeTokenServer:
    def __init__(self, replies=None):
        self.posts = []
        self.replies = list(replies or [])

    def __call__(self, url, data, as_json=False):
        self.posts.append((url, dict(data), as_json))
        if url.endswith("/register"):
            return {"client_id": "client-1", "redirect_uris": data["redirect_uris"]}
        return self.replies.pop(0)


def test_discovery_reads_satie_metadata_and_asks_for_a_refresh_token():
    server = satie_auth.discover("https://satie.example/mcp", fake_get(METADATA))
    assert server.resource == "https://satie.example/mcp"
    assert server.token_endpoint == AS + "/oauth/token"
    assert server.registration_endpoint == AS + "/oauth/clients/register"
    assert server.scopes == ("email", "profile", "offline_access")


def test_discovery_falls_back_to_the_root_well_known():
    table = {"https://satie.example/.well-known/oauth-protected-resource":
             METADATA["https://satie.example/.well-known/oauth-protected-resource/mcp"],
             "https://auth.example/auth/v1/.well-known/openid-configuration":
             METADATA["https://auth.example/.well-known/oauth-authorization-server/auth/v1"]}
    assert satie_auth.discover("https://satie.example/mcp", fake_get(table)).issuer == AS


def test_pkce_challenge_is_s256_of_the_verifier():
    verifier, challenge = satie_auth.pkce_pair()
    assert challenge == base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    assert 43 <= len(verifier) <= 128


def test_redirect_parsing_checks_state_and_errors():
    assert satie_auth.parse_redirect("http://localhost:1/callback?code=abc&state=s1", "s1") == "abc"
    with pytest.raises(SatieAuthError, match="state"):
        satie_auth.parse_redirect("http://localhost:1/callback?code=abc&state=other", "s1")
    with pytest.raises(SatieAuthError, match="access_denied"):
        satie_auth.parse_redirect("http://localhost:1/callback?error=access_denied&state=s1", "s1")


def _query(url):
    return {k: v[0] for k, v in urllib.parse.parse_qs(urllib.parse.urlsplit(url).query).items()}


def test_paste_login_registers_a_public_client_and_saves_the_refresh_token(tmp_path):
    captured = {}

    def say(text):
        if "?" in text and "http" in text:
            captured["url"] = text.split()[-1]

    def paste(prompt):
        q = _query(captured["url"])
        return f"{q['redirect_uri']}?code=code-1&state={q['state']}"

    tokens = FakeTokenServer([{"access_token": "at-1", "refresh_token": "rt-1", "expires_in": 3600}])
    path = tmp_path / "tok.json"
    satie_auth.login("https://satie.example/mcp", path=path, paste=True, say=say, ask=paste,
                     get_json=fake_get(METADATA), post=tokens)

    register = tokens.posts[0]
    assert register[2] is True and register[1]["token_endpoint_auth_method"] == "none"
    assert register[1]["grant_types"] == ["authorization_code", "refresh_token"]
    q = _query(captured["url"])
    assert q["code_challenge_method"] == "S256" and q["resource"] == "https://satie.example/mcp"
    assert q["scope"] == "email profile offline_access" and q["client_id"] == "client-1"

    exchange = tokens.posts[1][1]
    assert exchange["grant_type"] == "authorization_code" and exchange["code"] == "code-1"
    assert hashlib.sha256(exchange["code_verifier"].encode()).digest() == base64.urlsafe_b64decode(
        q["code_challenge"] + "=" * (-len(q["code_challenge"]) % 4))
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["client_id"] == "client-1" and saved["refresh_token"] == "rt-1"
    assert saved["token_endpoint"] == AS + "/oauth/token"


def test_browser_login_catches_the_localhost_redirect(tmp_path):
    tokens = FakeTokenServer([{"access_token": "at-1", "refresh_token": "rt-1", "expires_in": 3600}])

    def say(text):
        if "?" in text and "http" in text:  # play the browser: follow the redirect back
            q = _query(text.split()[-1])
            back = f"{q['redirect_uri']}?code=code-2&state={q['state']}"
            back = back.replace("localhost", "127.0.0.1")
            threading.Thread(target=lambda: urllib.request.urlopen(back, timeout=10).read(), daemon=True).start()

    path = tmp_path / "tok.json"
    satie_auth.login("https://satie.example/mcp", path=path, open_browser=False, timeout_s=20, say=say,
                     get_json=fake_get(METADATA), post=tokens)
    assert tokens.posts[1][1]["code"] == "code-2"
    assert json.loads(path.read_text(encoding="utf-8"))["access_token"] == "at-1"


def _token_file(tmp_path, expires_at, refresh="rt-1"):
    path = tmp_path / "tok.json"
    satie_auth.write_token_file(path, {"client_id": "client-1", "token_endpoint": AS + "/oauth/token",
                                       "resource": "https://satie.example/mcp", "access_token": "at-1",
                                       "refresh_token": refresh, "expires_at": expires_at})
    return path


def test_a_fresh_token_is_used_without_refreshing(tmp_path):
    tokens = FakeTokenServer()
    auth = SatieTokenAuth(_token_file(tmp_path, 10_000), post=tokens, now=lambda: 1_000)
    assert auth.access_token() == "at-1" and tokens.posts == []


def test_an_expiring_token_is_refreshed_and_the_rotated_refresh_token_saved(tmp_path):
    path = _token_file(tmp_path, 1_100)  # inside the 120 s margin
    tokens = FakeTokenServer([{"access_token": "at-2", "refresh_token": "rt-2", "expires_in": 3600}])
    auth = SatieTokenAuth(path, post=tokens, now=lambda: 1_000)
    assert auth.access_token() == "at-2"
    form = tokens.posts[0][1]
    assert form == {"grant_type": "refresh_token", "refresh_token": "rt-1", "client_id": "client-1",
                    "resource": "https://satie.example/mcp"}
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["refresh_token"] == "rt-2" and saved["expires_at"] == 4_600
    assert not path.with_name(path.name + ".lock").exists()
    # a server that doesn't rotate keeps the old refresh token
    tokens.replies.append({"access_token": "at-3", "expires_in": 60})
    SatieTokenAuth(path, post=tokens, now=lambda: 10_000).access_token()
    assert json.loads(path.read_text(encoding="utf-8"))["refresh_token"] == "rt-2"


def test_a_revoked_login_says_to_sign_in_again(tmp_path):
    def revoked(url, data, as_json=False):
        raise SatieAuthError(f"{url} answered 400 invalid_grant: Invalid Refresh Token")

    auth = SatieTokenAuth(_token_file(tmp_path, 0), post=revoked, now=lambda: 1_000)
    with pytest.raises(SatieAuthError, match="satie-login") as err:
        auth.access_token()
    assert "rt-1" not in str(err.value)
    with pytest.raises(SatieAuthError, match="satie-login"):
        SatieTokenAuth(tmp_path / "missing.json").access_token()


def test_worker_bearer_prefers_the_env_token_then_the_saved_login(tmp_path, monkeypatch):
    monkeypatch.setenv("SATIE_TOKEN_FILE", str(tmp_path / "none.json"))
    monkeypatch.delenv("SATIE_MCP_TOKEN", raising=False)
    assert satie_auth.bearer_from_env() is None
    path = _token_file(tmp_path, 4_102_444_800)  # 2100
    monkeypatch.setenv("SATIE_TOKEN_FILE", str(path))
    bearer = satie_auth.bearer_from_env()
    assert HttpMcpTransport("https://satie.example/mcp", bearer).headers == {"Authorization": "Bearer at-1"}
    monkeypatch.setenv("SATIE_MCP_TOKEN", "fixed")
    assert satie_auth.bearer_from_env() == "fixed"


def test_status_never_shows_tokens(tmp_path):
    info = satie_auth.status(_token_file(tmp_path, 1_600), now=lambda: 1_000)
    assert info["signed_in"] and info["has_refresh_token"] and info["access_token_expires_in_s"] == 600
    assert "at-1" not in json.dumps(info) and "rt-1" not in json.dumps(info)

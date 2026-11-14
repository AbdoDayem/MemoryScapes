"""Sign a headless worker in to SATIE once, then keep its access token fresh.

SATIE's MCP server takes OAuth 2.1 bearer tokens (its authorization server is Supabase
Auth, found through the server's protected-resource metadata). A worker can't click
through a browser on every run, so:

    memoryscapes-worker satie-login            # once, on the server, with its own SATIE account
    memoryscapes-worker satie-login --paste    # same, when the browser is on another machine
    memoryscapes-worker satie-login --status   # signed in? token expiry, plan and balance

The login registers a public client (dynamic client registration, no secret), runs the
authorization-code flow with PKCE, and saves the client id and refresh token to a
git-ignored file: ``<repo>/.satie/worker-token.json`` (``SATIE_TOKEN_FILE`` overrides).
Each SATIE call then uses the saved access token, refreshing it when it's about to
expire and saving the rotated refresh token. Tokens are never printed or logged.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import http.server
import json
import os
import secrets
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_TOKEN_FILE = REPO_ROOT / ".satie" / "worker-token.json"
DEFAULT_MCP_URL = "https://www.satie.world/mcp"
CLIENT_NAME = "MemoryScapes worker"
_UA = {"User-Agent": "memoryscapes-worker/1.0"}


class SatieAuthError(RuntimeError):
    pass


def token_file() -> Path:
    return Path(os.environ.get("SATIE_TOKEN_FILE") or DEFAULT_TOKEN_FILE)


# ---------------------------------------------------------------- HTTP (injectable)


def http_get_json(url: str) -> dict[str, Any]:
    req = urllib.request.Request(url, headers=dict(_UA, Accept="application/json"))
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 (SATIE's own URLs)
        return json.loads(resp.read())


def http_post(url: str, data: dict[str, Any], as_json: bool = False) -> dict[str, Any]:
    """POST a form (or JSON); returns the JSON reply. OAuth errors become SatieAuthError
    with the server's error code, never the request body (it holds tokens)."""
    body = json.dumps(data).encode() if as_json else urllib.parse.urlencode(data).encode()
    ctype = "application/json" if as_json else "application/x-www-form-urlencoded"
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers=dict(_UA, Accept="application/json", **{"Content-Type": ctype}))
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        try:
            err = json.loads(exc.read())
            detail = err.get("error_description") or err.get("error") or err.get("msg") or ""
            code = err.get("error") or err.get("error_code") or ""
        except (json.JSONDecodeError, ValueError, AttributeError):
            detail, code = "", ""
        raise SatieAuthError(f"{url} answered {exc.code} {code}: {detail}".rstrip(": ")) from None


# ---------------------------------------------------------------- discovery


@dataclass(frozen=True)
class AuthServer:
    resource: str
    issuer: str
    authorization_endpoint: str
    token_endpoint: str
    registration_endpoint: str | None
    scopes: tuple[str, ...]


def _well_known(base: str, name: str) -> list[str]:
    """RFC 8414 / RFC 9728 locations: the well-known segment goes between the origin and
    the path; older servers put it after the path."""
    u = urllib.parse.urlsplit(base)
    origin, path = f"{u.scheme}://{u.netloc}", u.path.rstrip("/")
    urls = [f"{origin}/.well-known/{name}{path}"]
    if path:
        urls += [f"{origin}{path}/.well-known/{name}", f"{origin}/.well-known/{name}"]
    return urls


def _first_json(urls: list[str], get_json: Callable[[str], dict[str, Any]]) -> dict[str, Any]:
    errors = []
    for url in urls:
        try:
            return get_json(url)
        except Exception as exc:  # noqa: BLE001 (try the next location)
            errors.append(f"{url}: {exc}")
    raise SatieAuthError("no OAuth metadata found (" + "; ".join(errors) + ")")


def discover(mcp_url: str, get_json: Callable[[str], dict[str, Any]] = http_get_json) -> AuthServer:
    pr = _first_json(_well_known(mcp_url, "oauth-protected-resource"), get_json)
    servers = pr.get("authorization_servers") or []
    if not servers:
        raise SatieAuthError("SATIE's metadata names no authorization server")
    issuer = servers[0].rstrip("/")
    meta = _first_json(_well_known(issuer, "oauth-authorization-server")
                       + _well_known(issuer, "openid-configuration"), get_json)
    supported = set(meta.get("scopes_supported") or [])
    scopes = list(pr.get("scopes_supported") or [])
    if "offline_access" in supported:  # ask for a refresh token explicitly
        scopes.append("offline_access")
    if "S256" not in (meta.get("code_challenge_methods_supported") or ["S256"]):
        raise SatieAuthError("SATIE's authorization server doesn't support PKCE S256")
    return AuthServer(
        resource=pr.get("resource") or mcp_url,
        issuer=issuer,
        authorization_endpoint=meta["authorization_endpoint"],
        token_endpoint=meta["token_endpoint"],
        registration_endpoint=meta.get("registration_endpoint"),
        scopes=tuple(dict.fromkeys(scopes)),
    )


# ---------------------------------------------------------------- token file


def _jwt_exp(token: str) -> float | None:
    """The `exp` claim of a JWT access token (read only for timing, not trusted)."""
    try:
        payload = token.split(".")[1]
        claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
        return float(claims["exp"])
    except (IndexError, KeyError, ValueError, TypeError):
        return None


def _expires_at(reply: dict[str, Any], now: float) -> float:
    if reply.get("expires_in") is not None:
        return now + float(reply["expires_in"])
    return _jwt_exp(reply["access_token"]) or now + 3600.0


def write_token_file(path: Path, data: dict[str, Any]) -> None:
    """Atomic replace, readable by this user only where the OS supports it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".token-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        with contextlib.suppress(OSError):  # no POSIX modes on Windows; .satie/ is per-user there
            os.chmod(tmp, 0o600)
        for attempt in range(20):  # Windows: a reader can hold the file open for a moment
            try:
                os.replace(tmp, path)
                break
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.05)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)


@contextlib.contextmanager
def _file_lock(path: Path, timeout_s: float = 30.0, stale_s: float = 60.0) -> Iterator[None]:
    """Cross-process lock (an exclusively created file) so two workers don't refresh at
    once: the second would present a refresh token the first just rotated away."""
    lock = path.with_name(path.name + ".lock")
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            with contextlib.suppress(FileNotFoundError):
                if time.time() - lock.stat().st_mtime > stale_s:  # a crashed holder
                    lock.unlink()
                    continue
            if time.monotonic() > deadline:
                raise SatieAuthError(f"timed out waiting for {lock}") from None
            time.sleep(0.1)
    try:
        yield
    finally:
        with contextlib.suppress(FileNotFoundError):
            lock.unlink()


class SatieTokenAuth:
    """Bearer tokens from the saved login, refreshed before they expire."""

    def __init__(self, path: Path, post: Callable[..., dict[str, Any]] = http_post,
                 now: Callable[[], float] = time.time, margin_s: float = 120.0) -> None:
        self.path = Path(path)
        self.post = post
        self.now = now
        self.margin_s = margin_s
        self._mem_lock = threading.Lock()

    def _read(self) -> dict[str, Any]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise SatieAuthError(f"not signed in to SATIE ({self.path} missing): run memoryscapes-worker satie-login") from None
        if not data.get("refresh_token") and not data.get("access_token"):
            raise SatieAuthError(f"{self.path} holds no token: run memoryscapes-worker satie-login")
        return data

    def _fresh(self, data: dict[str, Any]) -> bool:
        return bool(data.get("access_token")) and float(data.get("expires_at") or 0) - self.margin_s > self.now()

    def access_token(self) -> str:
        with self._mem_lock:
            data = self._read()
            if self._fresh(data):
                return data["access_token"]
            with _file_lock(self.path):
                data = self._read()  # another worker may have refreshed while we waited
                if not self._fresh(data):
                    data = self._refresh(data)
                    write_token_file(self.path, data)
            return data["access_token"]

    def _refresh(self, data: dict[str, Any]) -> dict[str, Any]:
        if not data.get("refresh_token"):
            raise SatieAuthError("SATIE access token expired and there's no refresh token: run memoryscapes-worker satie-login")
        form = {"grant_type": "refresh_token", "refresh_token": data["refresh_token"],
                "client_id": data["client_id"]}
        if data.get("client_secret"):
            form["client_secret"] = data["client_secret"]
        if data.get("resource"):
            form["resource"] = data["resource"]
        try:
            reply = self.post(data["token_endpoint"], form)
        except SatieAuthError as exc:
            raise SatieAuthError(f"SATIE sign-in could not be refreshed ({exc}); run memoryscapes-worker satie-login again") from None
        if not reply.get("access_token"):
            raise SatieAuthError("SATIE's token endpoint returned no access token")
        now = self.now()
        out = dict(data)
        out.update(access_token=reply["access_token"], expires_at=_expires_at(reply, now), refreshed_at=now)
        if reply.get("refresh_token"):  # rotated: the old one is no longer valid
            out["refresh_token"] = reply["refresh_token"]
        return out


def bearer_from_env() -> str | Callable[[], str] | None:
    """What the worker authenticates SATIE calls with: SATIE_MCP_TOKEN if set (a fixed
    token, e.g. for a quick test), else the saved login, else nothing."""
    if os.environ.get("SATIE_MCP_TOKEN"):
        return os.environ["SATIE_MCP_TOKEN"]
    path = token_file()
    if path.is_file():
        return SatieTokenAuth(path).access_token
    return None


# ---------------------------------------------------------------- login


def pkce_pair() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(48)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    return verifier, challenge


def register_client(server: AuthServer, redirect_uri: str,
                    post: Callable[..., dict[str, Any]] = http_post) -> dict[str, Any]:
    if not server.registration_endpoint:
        raise SatieAuthError("SATIE's authorization server has no client registration endpoint")
    reply = post(server.registration_endpoint, {
        "client_name": CLIENT_NAME,
        "redirect_uris": [redirect_uri],
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
        "token_endpoint_auth_method": "none",
        "scope": " ".join(server.scopes),
    }, as_json=True)
    if not reply.get("client_id"):
        raise SatieAuthError("client registration returned no client_id")
    return reply


def authorize_url(server: AuthServer, client_id: str, redirect_uri: str, state: str, challenge: str) -> str:
    query = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "scope": " ".join(server.scopes),
        "state": state,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "resource": server.resource,
    }
    return server.authorization_endpoint + "?" + urllib.parse.urlencode(query)


def parse_redirect(url: str, state: str) -> str:
    """The authorization code from the address the browser was sent back to."""
    query = urllib.parse.parse_qs(urllib.parse.urlsplit(url.strip()).query)
    if "error" in query:
        raise SatieAuthError("SATIE sign-in failed: " + (query.get("error_description") or query["error"])[0])
    if query.get("state", [None])[0] != state:
        raise SatieAuthError("the sign-in reply doesn't match this login (state mismatch); try again")
    code = query.get("code", [None])[0]
    if not code:
        raise SatieAuthError("no authorization code in that address")
    return code


def exchange_code(server: AuthServer, client: dict[str, Any], code: str, verifier: str, redirect_uri: str,
                  post: Callable[..., dict[str, Any]] = http_post, now: Callable[[], float] = time.time) -> dict[str, Any]:
    form = {"grant_type": "authorization_code", "code": code, "redirect_uri": redirect_uri,
            "client_id": client["client_id"], "code_verifier": verifier, "resource": server.resource}
    if client.get("client_secret"):
        form["client_secret"] = client["client_secret"]
    reply = post(server.token_endpoint, form)
    if not reply.get("access_token"):
        raise SatieAuthError("SATIE's token endpoint returned no access token")
    t = now()
    data = {
        "version": 1,
        "resource": server.resource,
        "token_endpoint": server.token_endpoint,
        "client_id": client["client_id"],
        "access_token": reply["access_token"],
        "refresh_token": reply.get("refresh_token"),
        "expires_at": _expires_at(reply, t),
        "scope": reply.get("scope") or " ".join(server.scopes),
        "signed_in_at": t,
    }
    if client.get("client_secret"):
        data["client_secret"] = client["client_secret"]
    return data


class _CallbackServer:
    """Waits for the browser's redirect to http://localhost:<port>/callback."""

    def __init__(self, port: int = 0) -> None:
        owner = self
        self.url: str | None = None

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                if not self.path.startswith("/callback"):
                    self.send_error(404)
                    return
                owner.url = "http://localhost" + self.path
                body = b"<!doctype html><title>MemoryScapes</title><p>Signed in to SATIE. You can close this tab.</p>"
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args: Any) -> None:  # keep the code out of the console
                pass

        self.httpd = http.server.HTTPServer(("127.0.0.1", port), Handler)
        self.port = self.httpd.server_address[1]

    def wait(self, timeout_s: float) -> str:
        deadline = time.monotonic() + timeout_s
        self.httpd.timeout = 1.0
        try:
            while self.url is None:
                if time.monotonic() > deadline:
                    raise SatieAuthError("timed out waiting for the browser; re-run, or use --paste")
                self.httpd.handle_request()
        finally:
            self.httpd.server_close()
        return self.url


def login(mcp_url: str = DEFAULT_MCP_URL, path: Path | None = None, paste: bool = False,
          open_browser: bool = True, timeout_s: float = 600.0,
          say: Callable[[str], None] = print, ask: Callable[[str], str] = input,
          get_json: Callable[[str], dict[str, Any]] | None = None,
          post: Callable[..., dict[str, Any]] | None = None) -> Path:
    """Interactive one-time sign-in; saves the token file and returns its path."""
    path = path or token_file()
    get_json, post = get_json or http_get_json, post or http_post
    server = discover(mcp_url, get_json)
    callback = None if paste else _CallbackServer()
    port = callback.port if callback else 8765  # --paste: nothing listens; the address is copied by hand
    redirect_uri = f"http://localhost:{port}/callback"
    client = register_client(server, redirect_uri, post)
    verifier, challenge = pkce_pair()
    state = secrets.token_urlsafe(16)
    url = authorize_url(server, client["client_id"], redirect_uri, state, challenge)

    say("Sign in to SATIE with the account this server should use:\n\n  " + url + "\n")
    if paste:
        say("After signing in, the browser goes to a localhost address that won't load.")
        code = parse_redirect(ask("Paste that whole address here: "), state)
    else:
        if open_browser:
            with contextlib.suppress(Exception):
                import webbrowser

                webbrowser.open(url)
        say("Waiting for the browser (if it can't reach localhost, re-run with --paste)...")
        code = parse_redirect(callback.wait(timeout_s), state)
    data = exchange_code(server, client, code, verifier, redirect_uri, post)
    data["mcp_url"] = mcp_url
    write_token_file(path, data)
    return path


def status(path: Path | None = None, now: Callable[[], float] = time.time) -> dict[str, Any]:
    """What's saved, without any token values."""
    path = path or token_file()
    if not path.is_file():
        return {"signed_in": False, "token_file": str(path)}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "signed_in": bool(data.get("refresh_token") or data.get("access_token")),
        "token_file": str(path),
        "client_id": data.get("client_id"),
        "has_refresh_token": bool(data.get("refresh_token")),
        "access_token_expires_in_s": round(float(data.get("expires_at") or 0) - now()),
        "scope": data.get("scope"),
    }

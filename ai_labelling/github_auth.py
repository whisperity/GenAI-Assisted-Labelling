"""GitHub App authentication helpers."""

import base64
import json
import os
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from typing import Optional

from ai_labelling.terminal import debug_log


_GITHUB_API = "https://api.github.com"


def _b64url(data: bytes) -> str:
    """Base64url-encode bytes, stripping ``=`` padding."""

    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _generate_jwt(app_id: str, private_key: str) -> str:
    """Create a signed RS256 JWT for GitHub App API authentication.

    Uses the system ``openssl`` binary for RSA signing — no extra Python
    packages are required.
    """

    # Environment variables often store PEM keys with literal \\n sequences
    # rather than real newlines.
    private_key = private_key.replace("\\n", "\n")

    now = int(time.time())
    header = _b64url(
        json.dumps({"alg": "RS256", "typ": "JWT"}).encode()
    )
    payload = _b64url(
        json.dumps({
            "iat": now - 60,   # allow minor clock skew
            "exp": now + 600,  # 10-minute maximum
            "iss": str(app_id),
        }).encode()
    )
    signing_input = f"{header}.{payload}".encode()

    fd, key_path = tempfile.mkstemp(suffix=".pem")
    try:
        os.write(fd, private_key.encode())
        os.close(fd)
        result = subprocess.run(
            ("openssl", "dgst", "-sha256", "-sign", key_path),
            input=signing_input,
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "openssl JWT signing failed: "
            + exc.stderr.decode("utf-8", errors="replace").strip()
        ) from exc
    except FileNotFoundError as exc:
        raise RuntimeError(
            "openssl not found; it is required for GitHub App authentication"
        ) from exc
    finally:
        os.unlink(key_path)

    return f"{header}.{payload}.{_b64url(result.stdout)}"


def _api_request(path: str, *, jwt: str, method: str = "GET") -> object:
    """Make one GitHub API request using a JWT bearer token."""

    url = _GITHUB_API + path
    req = urllib.request.Request(
        url,
        method=method,
        headers={
            "Authorization": f"Bearer {jwt}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"GitHub API HTTP {exc.code} at {url}: {body}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"GitHub API request failed for {url}: {exc.reason}"
        ) from exc


def _installation_id_for_repo(jwt: str, repo: str) -> int:
    """Return the App installation ID that covers ``repo``."""

    data = _api_request(f"/repos/{repo}/installation", jwt=jwt)
    if not isinstance(data, dict):
        raise RuntimeError(
            f"unexpected installation payload for {repo!r}"
        )
    installation_id = data.get("id")
    if not isinstance(installation_id, int):
        raise RuntimeError(
            f"no installation found for {repo!r} — "
            "is the GitHub App installed on this repository or organisation?"
        )
    return installation_id


def _access_token_for_installation(jwt: str, installation_id: int) -> str:
    """Exchange a JWT for an installation access token (1-hour TTL)."""

    data = _api_request(
        f"/app/installations/{installation_id}/access_tokens",
        jwt=jwt,
        method="POST",
    )
    if not isinstance(data, dict):
        raise RuntimeError(
            "unexpected access-token response from GitHub"
        )
    token = data.get("token")
    if not isinstance(token, str) or not token:
        raise RuntimeError("access token missing from GitHub response")
    return token


def resolve_github_token(repo: Optional[str] = None) -> Optional[str]:
    """Return a GitHub installation access token derived from App credentials.

    Returns ``None`` when ``GH_TOKEN`` / ``GITHUB_TOKEN`` is already set
    (``gh`` handles those itself) or when no App credentials are configured.
    Raises ``RuntimeError`` when credentials are partially configured.

    Environment variables:

    ``GH_TOKEN`` / ``GITHUB_TOKEN``
        Direct bearer token — takes precedence; App credentials are ignored.
    ``GITHUB_APP_ID``
        Numeric App ID.  Required for App authentication.
    ``GITHUB_APP_PRIVATE_KEY``
        PEM private-key content (literal ``\\n`` sequences are normalised).
    ``GITHUB_APP_PRIVATE_KEY_PATH``
        Path to the PEM private-key file.  Used when
        ``GITHUB_APP_PRIVATE_KEY`` is not set.
    ``GITHUB_APP_INSTALLATION_ID``
        Optional installation ID.  When absent the installation is looked up
        from the target repository via the API.
    """

    if os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN"):
        return None

    app_id = os.environ.get("GITHUB_APP_ID")
    if not app_id:
        return None

    private_key = os.environ.get("GITHUB_APP_PRIVATE_KEY")
    if not private_key:
        key_path = os.environ.get("GITHUB_APP_PRIVATE_KEY_PATH")
        if key_path:
            with open(key_path, encoding="utf-8") as fobj:
                private_key = fobj.read()

    if not private_key:
        raise RuntimeError(
            "'GITHUB_APP_ID' is set but neither 'GITHUB_APP_PRIVATE_KEY' "
            "nor 'GITHUB_APP_PRIVATE_KEY_PATH' is configured"
        )

    debug_log(
        f"Authenticating as GitHub App {app_id!r}",
        colour="yellow",
    )
    jwt = _generate_jwt(app_id, private_key)

    installation_id_str = os.environ.get("GITHUB_APP_INSTALLATION_ID")
    if installation_id_str:
        installation_id = int(installation_id_str)
    elif repo:
        installation_id = _installation_id_for_repo(jwt, repo)
    else:
        raise RuntimeError(
            "'GITHUB_APP_ID' is set but no repository is known yet — "
            "set 'GITHUB_APP_INSTALLATION_ID' or pass --repository"
        )

    return _access_token_for_installation(jwt, installation_id)

"""Tests for GitHub App authentication helpers."""
# pylint: disable=missing-function-docstring

import base64
import io
import json
import subprocess
import unittest
import urllib.error
from unittest import mock

from ai_labelling.github_auth import (
    _access_token_for_installation,
    _generate_jwt,
    _installation_id_for_repo,
    resolve_github_token,
)


def _make_urlopen_mock(payload: object):
    """Return a mock that simulates urllib.request.urlopen as a context mgr."""

    cm = mock.MagicMock()
    cm.__enter__ = mock.Mock(return_value=cm)
    cm.__exit__ = mock.Mock(return_value=False)
    cm.read.return_value = json.dumps(payload).encode()
    return cm


class GenerateJwtTests(unittest.TestCase):
    """Verify JWT structure and openssl error handling."""

    def _mock_run(self, stdout: bytes = b"fakesig"):
        return mock.patch(
            "ai_labelling.github_auth.subprocess.run",
            return_value=mock.Mock(stdout=stdout, returncode=0),
        )

    def test_jwt_has_three_parts(self):
        with self._mock_run():
            jwt = _generate_jwt("12345", "-----BEGIN RSA PRIVATE KEY-----\n")
        self.assertEqual(len(jwt.split(".")), 3)

    def test_jwt_header_is_rs256(self):
        with self._mock_run():
            jwt = _generate_jwt("12345", "KEY\n")
        header_b64 = jwt.split(".", maxsplit=1)[0]
        pad = "=" * (-len(header_b64) % 4)
        header = json.loads(base64.urlsafe_b64decode(header_b64 + pad))
        self.assertEqual(header["alg"], "RS256")
        self.assertEqual(header["typ"], "JWT")

    def test_jwt_payload_contains_app_id(self):
        with self._mock_run():
            jwt = _generate_jwt("99999", "KEY\n")
        parts = jwt.split(".")
        pad = "=" * (-len(parts[1]) % 4)
        payload = json.loads(base64.urlsafe_b64decode(parts[1] + pad))
        self.assertEqual(payload["iss"], "99999")

    def test_raises_when_openssl_missing(self):
        with mock.patch(
            "ai_labelling.github_auth.subprocess.run",
            side_effect=FileNotFoundError,
        ):
            with self.assertRaisesRegex(RuntimeError, "openssl not found"):
                _generate_jwt("1", "KEY\n")

    def test_raises_when_openssl_fails(self):
        exc = subprocess.CalledProcessError(1, "openssl", stderr=b"bad key")
        with mock.patch(
            "ai_labelling.github_auth.subprocess.run",
            side_effect=exc,
        ):
            with self.assertRaisesRegex(RuntimeError, "JWT signing failed"):
                _generate_jwt("1", "KEY\n")

    def test_normalises_literal_backslash_n_in_key(self):
        calls = []

        def fake_run(_argv, *, input, **_kw):  # pylint: disable=W0622
            calls.append(input)
            return mock.Mock(stdout=b"sig", returncode=0)

        with mock.patch("ai_labelling.github_auth.subprocess.run", fake_run), \
                mock.patch("ai_labelling.github_auth.os.unlink"):
            _generate_jwt("1", "-----BEGIN\\nKEY\\n-----END")

        self.assertGreater(len(calls), 0)

    def test_key_temp_file_cleaned_up_on_error(self):
        unlinked = []
        exc = FileNotFoundError("openssl")

        with mock.patch(
            "ai_labelling.github_auth.subprocess.run",
            side_effect=exc,
        ), mock.patch(
            "ai_labelling.github_auth.os.unlink",
            side_effect=unlinked.append,
        ):
            with self.assertRaises(RuntimeError):
                _generate_jwt("1", "KEY\n")

        self.assertEqual(len(unlinked), 1)


class InstallationIdTests(unittest.TestCase):
    """Verify repo-to-installation-ID lookup."""

    def test_returns_id_from_payload(self):
        cm = _make_urlopen_mock({"id": 42})
        with mock.patch("urllib.request.urlopen", return_value=cm):
            result = _installation_id_for_repo("jwt", "owner/repo")
        self.assertEqual(result, 42)

    def test_raises_when_id_missing(self):
        cm = _make_urlopen_mock({"not_id": "nope"})
        with mock.patch("urllib.request.urlopen", return_value=cm):
            with self.assertRaisesRegex(RuntimeError, "no installation found"):
                _installation_id_for_repo("jwt", "owner/repo")

    def test_raises_on_http_error(self):
        exc = urllib.error.HTTPError(
            "url", 404, "Not Found", {}, io.BytesIO(b"not found")
        )
        try:
            with mock.patch("urllib.request.urlopen", side_effect=exc):
                with self.assertRaisesRegex(RuntimeError, "HTTP 404"):
                    _installation_id_for_repo("jwt", "owner/repo")
        finally:
            exc.close()


class AccessTokenTests(unittest.TestCase):
    """Verify installation-token exchange."""

    def test_returns_token_string(self):
        cm = _make_urlopen_mock({"token": "ghs_abc123"})
        with mock.patch("urllib.request.urlopen", return_value=cm):
            token = _access_token_for_installation("jwt", 99)
        self.assertEqual(token, "ghs_abc123")

    def test_raises_when_token_missing(self):
        cm = _make_urlopen_mock({"expires_at": "..."})
        with mock.patch("urllib.request.urlopen", return_value=cm):
            with self.assertRaisesRegex(RuntimeError, "missing"):
                _access_token_for_installation("jwt", 99)


class ResolveGithubTokenTests(unittest.TestCase):
    """Verify env-variable priority and end-to-end token resolution."""

    def _env(self, **kwargs):
        return mock.patch.dict("os.environ", kwargs, clear=True)

    def test_returns_none_when_gh_token_set(self):
        with self._env(GH_TOKEN="existing"):
            result = resolve_github_token("owner/repo")
        self.assertIsNone(result)

    def test_returns_none_when_github_token_set(self):
        with self._env(GITHUB_TOKEN="existing"):
            result = resolve_github_token("owner/repo")
        self.assertIsNone(result)

    def test_returns_none_when_no_app_id(self):
        with self._env():
            result = resolve_github_token("owner/repo")
        self.assertIsNone(result)

    def test_raises_when_app_id_without_key(self):
        with self._env(GITHUB_APP_ID="1"):
            with self.assertRaisesRegex(
                RuntimeError, "GITHUB_APP_PRIVATE_KEY"
            ):
                resolve_github_token("owner/repo")

    def test_reads_key_from_path(self):
        pem = "-----BEGIN RSA PRIVATE KEY-----\n"
        env = {
            "GITHUB_APP_ID": "1",
            "GITHUB_APP_INSTALLATION_ID": "7",
            "GITHUB_APP_PRIVATE_KEY_PATH": "/tmp/key.pem",
        }
        with mock.patch.dict("os.environ", env, clear=True), \
                mock.patch("builtins.open", mock.mock_open(read_data=pem)), \
                mock.patch(
                    "ai_labelling.github_auth._generate_jwt",
                    return_value="jwt",
                ), mock.patch(
                    "ai_labelling.github_auth._access_token_for_installation",
                    return_value="ghs_tok",
                ):
            token = resolve_github_token("owner/repo")
        self.assertEqual(token, "ghs_tok")

    def test_uses_installation_id_env_when_set(self):
        with mock.patch.dict(
            "os.environ",
            {"GITHUB_APP_ID": "5",
             "GITHUB_APP_PRIVATE_KEY": "KEY",
             "GITHUB_APP_INSTALLATION_ID": "99"},
            clear=True,
        ), mock.patch(
            "ai_labelling.github_auth._generate_jwt",
            return_value="jwt",
        ), mock.patch(
            "ai_labelling.github_auth._installation_id_for_repo"
        ) as lookup_mock, mock.patch(
            "ai_labelling.github_auth._access_token_for_installation",
            return_value="ghs_tok",
        ) as token_mock:
            result = resolve_github_token("owner/repo")

        lookup_mock.assert_not_called()
        token_mock.assert_called_once_with("jwt", 99)
        self.assertEqual(result, "ghs_tok")

    def test_looks_up_installation_by_repo_when_id_not_set(self):
        with mock.patch.dict(
            "os.environ",
            {"GITHUB_APP_ID": "5",
             "GITHUB_APP_PRIVATE_KEY": "KEY"},
            clear=True,
        ), mock.patch(
            "ai_labelling.github_auth._generate_jwt",
            return_value="jwt",
        ), mock.patch(
            "ai_labelling.github_auth._installation_id_for_repo",
            return_value=77,
        ) as lookup_mock, mock.patch(
            "ai_labelling.github_auth._access_token_for_installation",
            return_value="ghs_tok",
        ):
            result = resolve_github_token("owner/repo")

        lookup_mock.assert_called_once_with("jwt", "owner/repo")
        self.assertEqual(result, "ghs_tok")

    def test_raises_when_no_repo_and_no_installation_id(self):
        with mock.patch.dict(
            "os.environ",
            {"GITHUB_APP_ID": "5", "GITHUB_APP_PRIVATE_KEY": "KEY"},
            clear=True,
        ), mock.patch(
            "ai_labelling.github_auth._generate_jwt",
            return_value="jwt",
        ):
            with self.assertRaisesRegex(
                RuntimeError, "no repository is known"
            ):
                resolve_github_token(None)

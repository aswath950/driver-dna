"""
test_drive_sync.py — Unit tests for src/drive_sync.py.

All tests use mocked Drive API clients and mocked google libraries; no real
network access or credentials are required.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_service(*, list_files=None, create_returns=None, update_returns=None):
    """Build a minimal mock Drive service."""
    svc = MagicMock()
    files_mock = svc.files.return_value

    list_exec = MagicMock(return_value={"files": list_files or []})
    files_mock.list.return_value.execute = list_exec

    create_result = create_returns or {"id": "new-file-id"}
    files_mock.create.return_value.execute = MagicMock(return_value=create_result)

    update_result = update_returns or {"id": "existing-file-id"}
    files_mock.update.return_value.execute = MagicMock(return_value=update_result)

    return svc


# ---------------------------------------------------------------------------
# is_authenticated
# ---------------------------------------------------------------------------

class TestIsAuthenticated:
    def test_returns_false_when_no_token_file(self, tmp_path):
        import drive_sync
        with patch.object(drive_sync, "_TOKEN_PATH", tmp_path / "no_token.json"):
            assert drive_sync.is_authenticated() is False

    def test_returns_true_when_valid_token_exists(self, tmp_path):
        import drive_sync
        mock_creds = MagicMock()
        mock_creds.expired = False

        with patch.object(drive_sync, "_load_token", return_value=mock_creds):
            assert drive_sync.is_authenticated() is True

    def test_returns_false_when_load_token_returns_none(self):
        import drive_sync
        with patch.object(drive_sync, "_load_token", return_value=None):
            assert drive_sync.is_authenticated() is False


# ---------------------------------------------------------------------------
# get_auth_url
# ---------------------------------------------------------------------------

class TestGetAuthUrl:
    def test_returns_none_when_client_config_missing(self):
        import drive_sync
        with patch.object(drive_sync, "_load_client_config", return_value=None):
            assert drive_sync.get_auth_url() is None

    def test_returns_url_string_when_config_present(self):
        import drive_sync
        cfg = {
            "client_id": "id.apps.googleusercontent.com",
            "client_secret": "secret",
            "redirect_uri": "http://localhost:8501",
        }
        mock_flow = MagicMock()
        mock_flow.authorization_url.return_value = ("https://accounts.google.com/o/oauth2/auth?...", "state")

        with patch.object(drive_sync, "_load_client_config", return_value=cfg), \
             patch("google_auth_oauthlib.flow.Flow") as mock_flow_cls:
            mock_flow_cls.from_client_config.return_value = mock_flow
            url = drive_sync.get_auth_url()

        assert url is not None
        assert url.startswith("https://")

    def test_returns_none_on_flow_exception(self):
        import drive_sync
        cfg = {"client_id": "x", "client_secret": "y", "redirect_uri": "http://localhost:8501"}

        with patch.object(drive_sync, "_load_client_config", return_value=cfg), \
             patch("google_auth_oauthlib.flow.Flow") as mock_flow_cls:
            mock_flow_cls.from_client_config.side_effect = Exception("boom")
            assert drive_sync.get_auth_url() is None


# ---------------------------------------------------------------------------
# exchange_code
# ---------------------------------------------------------------------------

class TestExchangeCode:
    def test_returns_false_when_client_config_missing(self):
        import drive_sync
        with patch.object(drive_sync, "_load_client_config", return_value=None):
            assert drive_sync.exchange_code("some-code") is False

    def test_saves_token_and_returns_true(self, tmp_path):
        import drive_sync
        cfg = {"client_id": "x", "client_secret": "y", "redirect_uri": "http://localhost:8501"}
        mock_flow = MagicMock()
        mock_flow.credentials = MagicMock()

        with patch.object(drive_sync, "_load_client_config", return_value=cfg), \
             patch.object(drive_sync, "_save_token") as mock_save, \
             patch("google_auth_oauthlib.flow.Flow") as mock_flow_cls:
            mock_flow_cls.from_client_config.return_value = mock_flow
            result = drive_sync.exchange_code("auth-code-xyz")

        assert result is True
        mock_save.assert_called_once()   # called with the flow's credentials object

    def test_clears_service_cache_on_success(self, tmp_path):
        import drive_sync
        drive_sync._service_cache = MagicMock()
        drive_sync._folder_id_cache = "old-folder"

        cfg = {"client_id": "x", "client_secret": "y", "redirect_uri": "http://localhost:8501"}
        mock_flow = MagicMock()
        mock_flow.credentials = MagicMock()

        with patch.object(drive_sync, "_load_client_config", return_value=cfg), \
             patch.object(drive_sync, "_save_token"), \
             patch("google_auth_oauthlib.flow.Flow") as mock_flow_cls:
            mock_flow_cls.from_client_config.return_value = mock_flow
            drive_sync.exchange_code("auth-code-xyz")

        assert drive_sync._service_cache is None
        assert drive_sync._folder_id_cache is None

    def test_returns_false_on_fetch_token_exception(self):
        import drive_sync
        cfg = {"client_id": "x", "client_secret": "y", "redirect_uri": "http://localhost:8501"}
        mock_flow = MagicMock()
        mock_flow.fetch_token.side_effect = Exception("invalid_grant")

        with patch.object(drive_sync, "_load_client_config", return_value=cfg), \
             patch("google_auth_oauthlib.flow.Flow") as mock_flow_cls:
            mock_flow_cls.from_client_config.return_value = mock_flow
            result = drive_sync.exchange_code("bad-code")

        assert result is False


# ---------------------------------------------------------------------------
# upload_file
# ---------------------------------------------------------------------------

class TestUploadFile:
    def test_returns_none_when_service_unavailable(self, tmp_path):
        import drive_sync
        test_file = tmp_path / "test.parquet"
        test_file.write_bytes(b"data")

        with patch.object(drive_sync, "_get_drive_service", return_value=None):
            result = drive_sync.upload_file(test_file)

        assert result is None

    def test_returns_none_when_local_file_absent(self, tmp_path):
        import drive_sync
        missing = tmp_path / "missing.parquet"
        svc = _make_service()

        with patch.object(drive_sync, "_get_drive_service", return_value=svc), \
             patch.object(drive_sync, "_get_or_create_folder", return_value="folder-id"):
            result = drive_sync.upload_file(missing)

        assert result is None

    def test_calls_create_for_new_file(self, tmp_path):
        import drive_sync
        test_file = tmp_path / "dataset.parquet"
        test_file.write_bytes(b"parquet-data")

        svc = _make_service(list_files=[], create_returns={"id": "created-id"})
        mock_http = MagicMock()
        mock_http.MediaFileUpload.return_value = MagicMock()

        with patch.object(drive_sync, "_get_drive_service", return_value=svc), \
             patch.object(drive_sync, "_get_or_create_folder", return_value="folder-id"), \
             patch.dict(sys.modules, {"googleapiclient.http": mock_http}):
            result = drive_sync.upload_file(test_file)

        assert result == "created-id"
        svc.files().create.assert_called_once()

    def test_calls_update_for_existing_file(self, tmp_path):
        import drive_sync
        test_file = tmp_path / "dataset.parquet"
        test_file.write_bytes(b"parquet-data")

        existing = [{"id": "existing-id"}]
        svc = _make_service(list_files=existing)
        mock_http = MagicMock()
        mock_http.MediaFileUpload.return_value = MagicMock()

        with patch.object(drive_sync, "_get_drive_service", return_value=svc), \
             patch.object(drive_sync, "_get_or_create_folder", return_value="folder-id"), \
             patch.dict(sys.modules, {"googleapiclient.http": mock_http}):
            result = drive_sync.upload_file(test_file)

        svc.files().update.assert_called_once()
        svc.files().create.assert_not_called()

    def test_returns_none_and_does_not_raise_on_api_error(self, tmp_path):
        import drive_sync
        test_file = tmp_path / "model.joblib"
        test_file.write_bytes(b"model-bytes")

        svc = MagicMock()
        svc.files.return_value.list.return_value.execute.side_effect = Exception("quota exceeded")

        with patch.object(drive_sync, "_get_drive_service", return_value=svc), \
             patch.object(drive_sync, "_get_or_create_folder", return_value="folder-id"):
            result = drive_sync.upload_file(test_file)

        assert result is None


# ---------------------------------------------------------------------------
# download_file
# ---------------------------------------------------------------------------

class TestDownloadFile:
    def test_returns_false_when_service_unavailable(self, tmp_path):
        import drive_sync
        with patch.object(drive_sync, "_get_drive_service", return_value=None):
            result = drive_sync.download_file("dataset.parquet", tmp_path / "dataset.parquet")
        assert result is False

    def test_returns_false_when_file_not_in_drive(self, tmp_path):
        import drive_sync
        svc = _make_service(list_files=[])

        with patch.object(drive_sync, "_get_drive_service", return_value=svc), \
             patch.object(drive_sync, "_get_or_create_folder", return_value="folder-id"):
            result = drive_sync.download_file("missing.parquet", tmp_path / "missing.parquet")

        assert result is False

    def test_creates_parent_dirs_and_returns_true(self, tmp_path):
        import drive_sync
        dest = tmp_path / "sub" / "dir" / "dataset.parquet"

        svc = _make_service(list_files=[{"id": "file-abc"}])
        mock_downloader = MagicMock()
        mock_downloader.next_chunk.return_value = (None, True)
        mock_http = MagicMock()
        mock_http.MediaIoBaseDownload.return_value = mock_downloader

        with patch.object(drive_sync, "_get_drive_service", return_value=svc), \
             patch.object(drive_sync, "_get_or_create_folder", return_value="folder-id"), \
             patch.dict(sys.modules, {"googleapiclient.http": mock_http}), \
             patch("io.FileIO", MagicMock()):
            result = drive_sync.download_file("dataset.parquet", dest)

        assert result is True
        assert dest.parent.exists()

    def test_returns_false_and_does_not_raise_on_api_error(self, tmp_path):
        import drive_sync
        svc = MagicMock()
        svc.files.return_value.list.return_value.execute.side_effect = Exception("network error")

        with patch.object(drive_sync, "_get_drive_service", return_value=svc), \
             patch.object(drive_sync, "_get_or_create_folder", return_value="folder-id"):
            result = drive_sync.download_file("model.joblib", tmp_path / "model.joblib")

        assert result is False


# ---------------------------------------------------------------------------
# restore_missing_artifacts
# ---------------------------------------------------------------------------

class TestRestoreMissingArtifacts:
    def test_downloads_only_absent_files(self, tmp_path, monkeypatch):
        import drive_sync

        present_file = tmp_path / "present.parquet"
        present_file.write_bytes(b"exists")
        absent_file = tmp_path / "absent.joblib"

        fake_critical = [
            ("present.parquet", present_file, "application/octet-stream"),
            ("absent.joblib",   absent_file,  "application/octet-stream"),
        ]
        monkeypatch.setattr(drive_sync, "_CRITICAL_FILES", fake_critical)

        with patch.object(drive_sync, "download_file", return_value=True) as mock_dl:
            drive_sync.restore_missing_artifacts()

        mock_dl.assert_called_once_with("absent.joblib", absent_file)

    def test_does_not_raise_when_download_fails(self, tmp_path, monkeypatch):
        import drive_sync

        absent_file = tmp_path / "broken.joblib"
        monkeypatch.setattr(drive_sync, "_CRITICAL_FILES", [
            ("broken.joblib", absent_file, "application/octet-stream"),
        ])

        with patch.object(drive_sync, "download_file", side_effect=Exception("boom")):
            drive_sync.restore_missing_artifacts()

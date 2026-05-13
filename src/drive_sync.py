"""
drive_sync.py — Google Drive artifact persistence for Driver DNA.

Each deployer authenticates with their own Google account via OAuth 2.0.
Artifacts are stored in that user's Drive under driver-dna-artifacts/.

Authentication flow:
  1. Call get_auth_url() to get the Google consent URL.
  2. User opens the URL, approves, and is redirected back with ?code=XXX.
  3. Call exchange_code(code) to exchange for tokens; tokens are saved to
     data/.google_token.json (gitignored).
  4. Subsequent calls load and auto-refresh the token from disk — silent.

Client credentials (client_id, client_secret, redirect_uri) are read from:
  - GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET / GOOGLE_REDIRECT_URI env vars
  - st.secrets["google"]["client_id"] etc. (Streamlit secrets fallback)

Scope: drive.file — the app can only see files it created.
All Drive calls are wrapped in try/except; failures are logged to stderr and
never propagate to callers.
"""

from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path
from typing import Any

_SRC = Path(__file__).parent
_ROOT = _SRC.parent
DATA_DIR = _ROOT / "data"
MODELS_DIR = _ROOT / "models"

_CRITICAL_FILES: list[tuple[str, Path, str]] = [
    ("dataset.parquet",        DATA_DIR   / "dataset.parquet",        "application/octet-stream"),
    ("dataset_meta.json",      DATA_DIR   / "dataset_meta.json",      "application/json"),
    ("circuits.json",          DATA_DIR   / "circuits.json",          "application/json"),
    ("driver_dna_clf.joblib",  MODELS_DIR / "driver_dna_clf.joblib",  "application/octet-stream"),
    ("label_encoder.joblib",   MODELS_DIR / "label_encoder.joblib",   "application/octet-stream"),
    ("metrics.json",           MODELS_DIR / "metrics.json",           "application/json"),
    ("confusion_matrix.html",  MODELS_DIR / "confusion_matrix.html",  "text/html"),
    ("shap_importance.png",    MODELS_DIR / "shap_importance.png",    "image/png"),
    ("llm_audit.jsonl",        MODELS_DIR / "llm_audit.jsonl",        "application/jsonlines"),
]

_DRIVE_FOLDER_NAME = "driver-dna-artifacts"
_DRIVE_SCOPE = "https://www.googleapis.com/auth/drive.file"
_TOKEN_PATH = DATA_DIR / ".google_token.json"

_GOOGLE_TOKEN_URI = "https://oauth2.googleapis.com/token"
_GOOGLE_AUTH_URI  = "https://accounts.google.com/o/oauth2/auth"

# Module-level cache — cleared on exchange_code() so new tokens take effect.
_service_cache: Any = None
_folder_id_cache: str | None = None


# ── Credential helpers ────────────────────────────────────────────────────────

def _load_client_config() -> dict | None:
    """Read OAuth client credentials from env vars or Streamlit secrets."""
    def _env_or_secret(env_key: str, secret_key: str) -> str | None:
        val = os.environ.get(env_key)
        if val:
            return val
        try:
            import streamlit as st
            return st.secrets["google"][secret_key]
        except Exception:
            return None

    client_id     = _env_or_secret("GOOGLE_CLIENT_ID",     "client_id")
    client_secret = _env_or_secret("GOOGLE_CLIENT_SECRET", "client_secret")
    redirect_uri  = _env_or_secret("GOOGLE_REDIRECT_URI",  "redirect_uri")

    if not (client_id and client_secret and redirect_uri):
        return None
    return {"client_id": client_id, "client_secret": client_secret, "redirect_uri": redirect_uri}


def _save_token(creds: Any) -> None:
    """Persist OAuth credentials to _TOKEN_PATH as JSON."""
    try:
        _TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "token":         creds.token,
            "refresh_token": creds.refresh_token,
            "token_uri":     creds.token_uri,
            "client_id":     creds.client_id,
            "client_secret": creds.client_secret,
            "scopes":        list(creds.scopes) if creds.scopes else [_DRIVE_SCOPE],
        }
        _TOKEN_PATH.write_text(json.dumps(data))
    except Exception as exc:
        print(f"[drive_sync] failed to save token: {exc}", file=sys.stderr)


def _load_token() -> Any:
    """
    Load OAuth credentials from _TOKEN_PATH, refreshing if expired.
    Returns a valid Credentials object, or None if unavailable.
    """
    if not _TOKEN_PATH.exists():
        return None
    try:
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request

        data = json.loads(_TOKEN_PATH.read_text())
        creds = Credentials(
            token=data.get("token"),
            refresh_token=data.get("refresh_token"),
            token_uri=data.get("token_uri", _GOOGLE_TOKEN_URI),
            client_id=data.get("client_id"),
            client_secret=data.get("client_secret"),
            scopes=data.get("scopes", [_DRIVE_SCOPE]),
        )
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            _save_token(creds)
        return creds
    except Exception as exc:
        print(f"[drive_sync] token load/refresh failed: {exc}", file=sys.stderr)
        return None


# ── Public auth interface ─────────────────────────────────────────────────────

def is_authenticated() -> bool:
    """Return True if a valid (or refreshable) Drive token exists on disk."""
    return _load_token() is not None


def get_auth_url() -> str | None:
    """
    Build and return the Google OAuth consent URL.
    Returns None if client credentials are not configured.
    """
    try:
        cfg = _load_client_config()
        if cfg is None:
            return None

        from google_auth_oauthlib.flow import Flow
        flow = Flow.from_client_config(
            {"web": {
                "client_id":     cfg["client_id"],
                "client_secret": cfg["client_secret"],
                "auth_uri":      _GOOGLE_AUTH_URI,
                "token_uri":     _GOOGLE_TOKEN_URI,
            }},
            scopes=[_DRIVE_SCOPE],
        )
        flow.redirect_uri = cfg["redirect_uri"]
        auth_url, _ = flow.authorization_url(
            access_type="offline",
            include_granted_scopes="false",
            prompt="consent",
        )
        return auth_url
    except Exception as exc:
        print(f"[drive_sync] get_auth_url failed: {exc}", file=sys.stderr)
        return None


def exchange_code(code: str) -> bool:
    """
    Exchange an OAuth authorization code for tokens and persist them.
    Clears the cached Drive service so the next call uses the new token.
    Returns True on success, False on any failure.
    """
    global _service_cache, _folder_id_cache
    try:
        cfg = _load_client_config()
        if cfg is None:
            return False

        from google_auth_oauthlib.flow import Flow
        flow = Flow.from_client_config(
            {"web": {
                "client_id":     cfg["client_id"],
                "client_secret": cfg["client_secret"],
                "auth_uri":      _GOOGLE_AUTH_URI,
                "token_uri":     _GOOGLE_TOKEN_URI,
            }},
            scopes=[_DRIVE_SCOPE],
        )
        flow.redirect_uri = cfg["redirect_uri"]
        flow.fetch_token(code=code)
        _save_token(flow.credentials)
        _service_cache = None
        _folder_id_cache = None
        print("[drive_sync] OAuth token saved — Drive connected", file=sys.stderr)
        return True
    except Exception as exc:
        print(f"[drive_sync] exchange_code failed: {exc}", file=sys.stderr)
        return False


# ── Drive client ──────────────────────────────────────────────────────────────

def _get_drive_service() -> Any:
    """Return a cached Drive v3 API client, or None if not authenticated."""
    global _service_cache
    if _service_cache is not None:
        return _service_cache

    creds = _load_token()
    if creds is None:
        return None

    try:
        from googleapiclient.discovery import build
        _service_cache = build("drive", "v3", credentials=creds, cache_discovery=False)
        return _service_cache
    except Exception as exc:
        print(f"[drive_sync] Drive service build failed: {exc}", file=sys.stderr)
        return None


def _get_or_create_folder(service: Any) -> str | None:
    """Return the Drive folder ID for driver-dna-artifacts, creating it if needed."""
    global _folder_id_cache
    if _folder_id_cache is not None:
        return _folder_id_cache

    try:
        query = (
            f"name='{_DRIVE_FOLDER_NAME}' "
            f"and mimeType='application/vnd.google-apps.folder' "
            f"and trashed=false"
        )
        results = service.files().list(q=query, fields="files(id, name)", spaces="drive").execute()
        files = results.get("files", [])
        if files:
            _folder_id_cache = files[0]["id"]
            return _folder_id_cache

        folder_meta = {
            "name": _DRIVE_FOLDER_NAME,
            "mimeType": "application/vnd.google-apps.folder",
        }
        folder = service.files().create(body=folder_meta, fields="id").execute()
        _folder_id_cache = folder["id"]
        print(f"[drive_sync] created Drive folder '{_DRIVE_FOLDER_NAME}' ({_folder_id_cache})", file=sys.stderr)
        return _folder_id_cache
    except Exception as exc:
        print(f"[drive_sync] folder lookup/create failed: {exc}", file=sys.stderr)
        return None


# ── Public sync interface ─────────────────────────────────────────────────────

def upload_file(local_path: Path, mime_type: str = "application/octet-stream") -> str | None:
    """
    Upload local_path to the driver-dna-artifacts Drive folder.
    Updates in-place if the file already exists (avoids duplicates).
    Returns the Drive file ID on success, None on any failure.
    """
    try:
        from googleapiclient.http import MediaFileUpload

        service = _get_drive_service()
        if service is None:
            return None

        if not local_path.exists():
            print(f"[drive_sync] skipping upload — {local_path} does not exist", file=sys.stderr)
            return None

        folder_id = _get_or_create_folder(service)
        if folder_id is None:
            return None

        filename = local_path.name
        query = f"name='{filename}' and '{folder_id}' in parents and trashed=false"
        existing = service.files().list(q=query, fields="files(id)").execute().get("files", [])

        media = MediaFileUpload(str(local_path), mimetype=mime_type, resumable=False)
        if existing:
            file_id = existing[0]["id"]
            service.files().update(fileId=file_id, media_body=media).execute()
        else:
            meta = {"name": filename, "parents": [folder_id]}
            result = service.files().create(body=meta, media_body=media, fields="id").execute()
            file_id = result["id"]

        print(f"[drive_sync] uploaded {filename} → Drive file ID {file_id}", file=sys.stderr)
        return file_id
    except Exception as exc:
        print(f"[drive_sync] upload failed for {local_path.name}: {exc}", file=sys.stderr)
        return None


def download_file(drive_filename: str, local_path: Path) -> bool:
    """
    Download drive_filename from driver-dna-artifacts to local_path.
    Creates parent directories as needed.
    Returns True on success, False on any failure.
    """
    try:
        from googleapiclient.http import MediaIoBaseDownload

        service = _get_drive_service()
        if service is None:
            return False

        folder_id = _get_or_create_folder(service)
        if folder_id is None:
            return False

        query = f"name='{drive_filename}' and '{folder_id}' in parents and trashed=false"
        results = service.files().list(q=query, fields="files(id)").execute()
        files = results.get("files", [])
        if not files:
            print(f"[drive_sync] {drive_filename} not found in Drive folder", file=sys.stderr)
            return False

        file_id = files[0]["id"]
        local_path.parent.mkdir(parents=True, exist_ok=True)

        request = service.files().get_media(fileId=file_id)
        with io.FileIO(str(local_path), "wb") as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()

        print(f"[drive_sync] restored {drive_filename} from Drive → {local_path}", file=sys.stderr)
        return True
    except Exception as exc:
        print(f"[drive_sync] download failed for {drive_filename}: {exc}", file=sys.stderr)
        return False


def restore_missing_artifacts() -> None:
    """
    For each critical artifact that is absent locally, attempt a Drive restore.
    Called at app startup when the user is authenticated.
    """
    try:
        for filename, local_path, _ in _CRITICAL_FILES:
            if not local_path.exists():
                download_file(filename, local_path)
    except Exception as exc:
        print(f"[drive_sync] restore_missing_artifacts failed: {exc}", file=sys.stderr)

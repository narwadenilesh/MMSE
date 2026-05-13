import os
import tempfile
from typing import Optional
import requests
from dotenv import load_dotenv

load_dotenv()


def transcribe_audio_bytes(audio_bytes: bytes, language_code: Optional[str] = None) -> Optional[str]:
    """Transcribe uploaded audio bytes into text using Sarvam Speech-to-Text REST API."""
    api_key = os.getenv("SARVAM_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing SARVAM_API_KEY in .env file.")

    endpoint = os.getenv("SARVAM_STT_ENDPOINT", "https://api.sarvam.ai/speech-to-text")
    model = os.getenv("SARVAM_STT_MODEL", "saaras:v3")
    mode = os.getenv("SARVAM_STT_MODE", "transcribe")
    selected_language_code = language_code or os.getenv("SARVAM_STT_LANGUAGE_CODE", "").strip()

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        headers = {
            "api-subscription-key": api_key,
        }

        data = {
            "model": model,
            "mode": mode,
        }
        if selected_language_code:
            data["language_code"] = selected_language_code

        with open(tmp_path, "rb") as audio_file:
            files = {
                "file": (os.path.basename(tmp_path), audio_file, "audio/wav"),
            }
            response = requests.post(
                endpoint,
                headers=headers,
                data=data,
                files=files,
                timeout=90,
            )

        if response.status_code >= 400:
            raise RuntimeError(f"Sarvam STT error {response.status_code}: {response.text}")

        payload = response.json()
        text = str(payload.get("transcript", "")).strip()
        return text if text else None
    finally:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

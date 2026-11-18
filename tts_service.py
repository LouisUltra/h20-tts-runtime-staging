"""Small, lazy-loading Qwen3-TTS service for the H20 GPU middle layer.

The service keeps model weights on the mounted persistent disk and loads the
model on the first synthesis request.  Health endpoints deliberately report
artifact readiness separately from model-loaded state so an idle service does
not occupy GPU memory merely to satisfy the platform probe.
"""

from __future__ import annotations

import io
import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger("h20-tts")
MAX_TEXT_LENGTH = 4096
DEFAULT_MODEL_DIR = "/models/qwen3-tts-0.6b-customvoice"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8080


def _utc_seconds() -> float:
    return time.time()


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if parsed <= 0 or parsed > 65535:
        raise ValueError(f"{name} must be between 1 and 65535")
    return parsed


@dataclass(frozen=True)
class SynthesisRequest:
    text: str
    language: str
    speaker: str
    instruct: str | None

    @classmethod
    def from_payload(cls, payload: Any) -> "SynthesisRequest":
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        text = payload.get("input", payload.get("text"))
        if not isinstance(text, str) or not text.strip():
            raise ValueError("input must be a non-empty string")
        text = text.strip()
        if len(text) > MAX_TEXT_LENGTH:
            raise ValueError(f"input is limited to {MAX_TEXT_LENGTH} characters")

        language = payload.get("language", "Chinese")
        speaker = payload.get("voice", payload.get("speaker", "Vivian"))
        instruct = payload.get("instructions", payload.get("instruct"))
        if not isinstance(language, str) or not language.strip():
            raise ValueError("language must be a non-empty string")
        if not isinstance(speaker, str) or not speaker.strip():
            raise ValueError("voice must be a non-empty string")
        if instruct is not None and not isinstance(instruct, str):
            raise ValueError("instructions must be a string when provided")
        return cls(
            text=text,
            language=language.strip(),
            speaker=speaker.strip(),
            instruct=instruct.strip() if isinstance(instruct, str) and instruct.strip() else None,
        )


class TTSRuntime:
    """Owns the lazy model lifecycle and serializes GPU synthesis calls."""

    def __init__(self, model_dir: str | Path) -> None:
        self.model_dir = Path(model_dir).expanduser()
        self._model: Any = None
        self._load_error: str | None = None
        self._loaded_at: float | None = None
        self._last_request_at: float | None = None
        self._request_count = 0
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    def _artifact_ready(self) -> bool:
        return self.model_dir.is_dir() and (self.model_dir / "config.json").is_file()

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        with self._load_lock:
            if self._model is not None:
                return self._model
            if not self._artifact_ready():
                raise RuntimeError(f"TTS model artifact is not ready: {self.model_dir}")
            try:
                import torch
                from qwen_tts import Qwen3TTSModel

                use_cuda = torch.cuda.is_available()
                kwargs: dict[str, Any] = {
                    "device_map": "cuda:0" if use_cuda else "cpu",
                    "dtype": torch.bfloat16 if use_cuda else torch.float32,
                }
                attention = os.environ.get("QWEN_TTS_ATTN", "eager").strip()
                if attention:
                    kwargs["attn_implementation"] = attention
                LOGGER.info("loading TTS model from %s", self.model_dir)
                self._model = Qwen3TTSModel.from_pretrained(str(self.model_dir), **kwargs)
                self._loaded_at = _utc_seconds()
                self._load_error = None
                LOGGER.info("TTS model loaded, cuda=%s", use_cuda)
                return self._model
            except Exception as exc:
                self._load_error = f"{type(exc).__name__}: {exc}"
                LOGGER.exception("TTS model load failed")
                raise

    def synthesize(self, request: SynthesisRequest) -> tuple[bytes, int, str]:
        model = self._load()
        with self._inference_lock:
            try:
                wavs, sample_rate = model.generate_custom_voice(
                    text=request.text,
                    language=request.language,
                    speaker=request.speaker,
                    instruct=request.instruct,
                )
                if not wavs:
                    raise RuntimeError("TTS model returned no audio")
                import soundfile as sf

                output = io.BytesIO()
                sf.write(output, wavs[0], sample_rate, format="WAV")
                self._last_request_at = _utc_seconds()
                self._request_count += 1
                return output.getvalue(), int(sample_rate), str(uuid.uuid4())
            except Exception:
                LOGGER.exception("TTS synthesis failed")
                raise

    def health(self) -> dict[str, Any]:
        return {
            "service": "h20-tts",
            "status": "ready" if self._artifact_ready() else "waiting_for_artifact",
            "artifact_ready": self._artifact_ready(),
            "model_loaded": self._model is not None,
            "model_dir": str(self.model_dir),
            "loaded_at_epoch": self._loaded_at,
            "last_request_at_epoch": self._last_request_at,
            "request_count": self._request_count,
            "load_error": self._load_error,
            "health_evidence": "model_service_request" if self._request_count else "artifact_probe_only",
        }


class RequestHandler(BaseHTTPRequestHandler):
    runtime: TTSRuntime

    def log_message(self, fmt: str, *args: Any) -> None:
        LOGGER.info("%s - %s", self.address_string(), fmt % args)

    def _write_json(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self) -> Any:
        raw_length = self.headers.get("Content-Length")
        try:
            length = int(raw_length or "0")
        except ValueError as exc:
            raise ValueError("Content-Length must be an integer") from exc
        if length <= 0 or length > 64 * 1024:
            raise ValueError("request body must be between 1 and 65536 bytes")
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("request body must be valid UTF-8 JSON") from exc

    def do_GET(self) -> None:
        if self.path in {"/health", "/healthz", "/readyz", "/v1/health"}:
            health = self.runtime.health()
            status = HTTPStatus.OK if health["artifact_ready"] else HTTPStatus.SERVICE_UNAVAILABLE
            self._write_json(status, health)
            return
        if self.path == "/v1/models":
            self._write_json(
                HTTPStatus.OK,
                {"data": [{"id": self.runtime.model_dir.name, "object": "model"}]},
            )
            return
        self._write_json(HTTPStatus.NOT_FOUND, {"error": "route not found"})

    def do_POST(self) -> None:
        if self.path != "/v1/audio/speech":
            self._write_json(HTTPStatus.NOT_FOUND, {"error": "route not found"})
            return
        try:
            request = SynthesisRequest.from_payload(self._read_json())
            audio, sample_rate, request_id = self.runtime.synthesize(request)
        except ValueError as exc:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        except Exception as exc:
            self._write_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": "TTS synthesis failed", "detail": f"{type(exc).__name__}: {exc}"},
            )
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(audio)))
        self.send_header("X-Request-ID", request_id)
        self.send_header("X-Sample-Rate", str(sample_rate))
        self.end_headers()
        self.wfile.write(audio)


def build_server() -> ThreadingHTTPServer:
    model_dir = os.environ.get("QWEN_TTS_MODEL_DIR", DEFAULT_MODEL_DIR)
    host = os.environ.get("H20_TTS_HOST", DEFAULT_HOST)
    port = _env_int("H20_TTS_PORT", DEFAULT_PORT)
    runtime = TTSRuntime(model_dir)
    handler = type("TTSRequestHandler", (RequestHandler,), {"runtime": runtime})
    server = ThreadingHTTPServer((host, port), handler)
    LOGGER.info("serving H20 TTS on %s:%s with model_dir=%s", host, port, model_dir)
    return server


def main() -> int:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    server = build_server()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("shutdown requested")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

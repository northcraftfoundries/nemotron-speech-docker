import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.asr import (
    LANG_TO_ID,
    ModelNotReadyError,
    NemotronASR,
    SUPPORTED_CHUNK_MS,
    UnsupportedChunkSizeError,
    UnsupportedLanguageError,
    build_engine,
)
from app.config import AppConfig, load_config


logger = logging.getLogger("uvicorn.error")
settings: AppConfig = load_config()
engine: NemotronASR = build_engine(
    model_id=settings.model_id,
    model_dir=settings.model_dir,
    execution_provider=settings.execution_provider,
)
load_error: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global load_error
    try:
        logger.info("ASR startup: loading model_id=%s provider=%s", engine.model_id, engine.execution_provider)
        await asyncio.to_thread(engine.load)
        logger.info(
            "ASR ready: model_dir=%s sample_rate=%s chunk_samples=%s default_chunk_ms=%s",
            engine.model_dir,
            engine.sample_rate,
            engine.chunk_samples,
            settings.default_chunk_ms,
        )
        if settings.startup_self_test:
            await _run_startup_self_tests(settings)
    except Exception as exc:
        load_error = str(exc)
        logger.exception("ASR startup failed: %s", exc)
    yield


app = FastAPI(title="Nemotron 3.5 ASR ONNX INT4", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=600,
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
async def ready() -> JSONResponse:
    if engine.ready:
        return JSONResponse(
            {
                "status": "ready",
                "model_id": engine.model_id,
                "model_dir": str(engine.model_dir),
                "sample_rate": engine.sample_rate,
                "chunk_samples": engine.chunk_samples,
                "default_chunk_ms": settings.default_chunk_ms,
                "supported_chunk_ms": list(SUPPORTED_CHUNK_MS),
                "execution_provider": engine.execution_provider,
            }
        )
    return JSONResponse({"status": "not_ready", "error": load_error}, status_code=503)


@app.get("/v1/languages")
async def languages() -> dict[str, Any]:
    return {
        "languages": [
            {"code": code, "id": lang_id, "name": name}
            for code, (lang_id, name) in sorted(LANG_TO_ID.items())
        ]
    }


@app.post("/v1/transcriptions")
async def transcriptions(
    file: Annotated[UploadFile, File()],
    language: Annotated[str, Form()] = "auto",
    use_vad: Annotated[bool, Form()] = False,
    chunk_ms: Annotated[int, Form()] = settings.default_chunk_ms,
) -> dict[str, Any]:
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty")
    try:
        result = await engine.transcribe_bytes(
            content=content,
            filename=file.filename or "audio",
            language=language,
            use_vad=use_vad,
            chunk_ms=chunk_ms,
        )
    except UnsupportedLanguageError as exc:
        raise HTTPException(
            status_code=400,
            detail={"message": str(exc), "supported": sorted(LANG_TO_ID)},
        ) from exc
    except ModelNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except UnsupportedChunkSizeError as exc:
        raise HTTPException(
            status_code=400,
            detail={"message": str(exc), "supported_chunk_ms": list(SUPPORTED_CHUNK_MS)},
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not transcribe audio: {exc}") from exc

    return {
        "text": result.text,
        "language": result.language,
        "language_name": result.language_name,
        "duration_seconds": result.duration_seconds,
        "wall_seconds": result.wall_seconds,
        "rtf": result.rtf,
        "sample_rate": result.sample_rate,
        "chunk_ms": result.chunk_ms,
        "chunk_samples": result.chunk_samples,
        "chunks_total": result.chunks_total,
        "chunks_processed": result.chunks_processed,
        "chunks_skipped": result.chunks_skipped,
        "vad_enabled": result.vad_enabled,
    }


@app.websocket("/v1/transcriptions/stream")
async def transcription_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    stream = None
    language = "auto"
    use_vad = False
    chunk_ms = settings.default_chunk_ms

    try:
        first = await websocket.receive()
        if "text" in first and first["text"]:
            config = _parse_control(first["text"])
            language = str(config.get("language", language))
            use_vad = bool(config.get("use_vad", use_vad))
            chunk_ms = int(config.get("chunk_ms", chunk_ms))
            requested_sample_rate = config.get("sample_rate", engine.sample_rate)
            if requested_sample_rate != engine.sample_rate:
                await websocket.send_json(
                    {
                        "event": "error",
                        "message": f"sample_rate must be {engine.sample_rate}; resampling is only supported by the upload endpoint",
                    }
                )
                return
        elif "bytes" in first and first["bytes"] is not None:
            pass
        else:
            await websocket.send_json({"event": "error", "message": "Expected config JSON or binary PCM audio"})
            return

        stream = await engine.create_stream(language=language, use_vad=use_vad, chunk_ms=chunk_ms)
        await websocket.send_json(
            {
                "event": "ready",
                "sample_rate": engine.sample_rate,
                "chunk_ms": stream.chunk_ms,
                "chunk_samples": stream.chunk_samples,
                "supported_chunk_ms": list(SUPPORTED_CHUNK_MS),
                "language": language,
                "language_name": stream.language_name,
                "format": "f32le",
            }
        )

        if "bytes" in first and first["bytes"] is not None:
            partial = stream.accept_pcm_f32le(first["bytes"])
            if partial:
                await websocket.send_json({"event": "partial", "text": partial})

        while True:
            message = await websocket.receive()
            if "bytes" in message and message["bytes"] is not None:
                partial = stream.accept_pcm_f32le(message["bytes"])
                if partial:
                    await websocket.send_json({"event": "partial", "text": partial})
            elif "text" in message and message["text"]:
                control = _parse_control(message["text"])
                if control.get("event") == "end":
                    await websocket.send_json(stream.finish())
                    return
            elif message.get("type") == "websocket.disconnect":
                return
    except WebSocketDisconnect:
        return
    except UnsupportedLanguageError as exc:
        await websocket.send_json({"event": "error", "message": str(exc), "supported": sorted(LANG_TO_ID)})
    except ModelNotReadyError as exc:
        await websocket.send_json({"event": "error", "message": str(exc)})
    except UnsupportedChunkSizeError as exc:
        await websocket.send_json(
            {"event": "error", "message": str(exc), "supported_chunk_ms": list(SUPPORTED_CHUNK_MS)}
        )
    except Exception as exc:
        await websocket.send_json({"event": "error", "message": str(exc)})
    finally:
        if stream is not None:
            stream.close()


def _parse_control(text: str) -> dict[str, Any]:
    import json

    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("Text WebSocket messages must be JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("Text WebSocket messages must be JSON objects")
    return value


async def _run_startup_self_tests(config: AppConfig) -> None:
    if not config.self_test_audio:
        logger.info("ASR self-test: no startup audio configured")
        return

    logger.info("ASR self-test: running %s audio check(s)", len(config.self_test_audio))
    for test in config.self_test_audio:
        if not test.path.exists():
            logger.warning("ASR self-test skipped: language=%s file=%s missing", test.language, test.path)
            continue
        try:
            content = await asyncio.to_thread(test.path.read_bytes)
            result = await engine.transcribe_bytes(
                content=content,
                filename=test.path.name,
                language=test.language,
                use_vad=False,
                chunk_ms=config.default_chunk_ms,
            )
            preview = result.text[:120].replace("\n", " ")
            if len(result.text) > 120:
                preview += "..."
            logger.info(
                "ASR self-test ok: language=%s duration=%.2fs wall=%.2fs rtf=%s text=%r",
                result.language,
                result.duration_seconds,
                result.wall_seconds,
                f"{result.rtf:.2f}" if result.rtf is not None else "n/a",
                preview,
            )
        except Exception as exc:
            logger.exception("ASR self-test failed: language=%s file=%s error=%s", test.language, test.path, exc)

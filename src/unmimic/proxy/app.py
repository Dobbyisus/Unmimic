"""OpenAI-compatible proxy.

Every request passes the input guards, goes to the upstream model, and every response
passes the output guards on its way back. With no output guards configured, responses
are forwarded unchanged (streams byte for byte).
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import AsyncIterator, Iterable, Mapping, Sequence
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from unmimic.plugins import Context, InputGuard, OutputGuard
from unmimic.reasoning import DEFAULT_FORMAT, ModelFormat, StreamReasoning, split_message

log = logging.getLogger(__name__)

ANONYMOUS = "anonymous"
# Reasoning models can think for many minutes before replying (DeepSeek-R1 ~8k tokens at
# ~20 tok/s per request on one GPU), so the default read timeout is an hour.
DEFAULT_UPSTREAM_TIMEOUT = 3600.0


def account_id_from(request: Request) -> str:
    """Derive a stable account id from the caller's API key without keeping the key."""
    scheme, _, key = request.headers.get("authorization", "").partition(" ")
    key = key.strip()
    if scheme.lower() != "bearer" or not key:
        return ANONYMOUS
    return "acct_" + hashlib.sha256(key.encode()).hexdigest()[:16]


def error_response(status: int, message: str, error_type: str) -> JSONResponse:
    """An error in the OpenAI shape, so clients surface it normally."""
    error = {"message": message, "type": error_type, "code": error_type}
    return JSONResponse(status_code=status, content={"error": error})


def upstream_unreachable(exc: httpx.HTTPError) -> JSONResponse:
    message = f"upstream model unreachable ({type(exc).__name__})"
    return error_response(502, message, "upstream_error")


def passthrough(upstream: httpx.Response) -> Response:
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type"),
    )


def create_app(
    upstream_url: str,
    input_guards: Iterable[InputGuard] = (),
    output_guards: Iterable[OutputGuard] = (),
    upstream_api_key: str | None = None,
    formats: Mapping[str, ModelFormat] | None = None,
    upstream_timeout: float = DEFAULT_UPSTREAM_TIMEOUT,
    transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    """Build the proxy.

    ``formats`` maps model names to where each puts its reasoning (written by
    ``unmimic init``); unknown models use ``DEFAULT_FORMAT``. ``upstream_timeout`` is how
    long to wait for the model's reply, in seconds. ``transport`` replaces the network, for
    tests.
    """
    inputs = list(input_guards)
    outputs = list(output_guards)
    known_formats = dict(formats or {})
    headers = {"authorization": f"Bearer {upstream_api_key}"} if upstream_api_key else {}

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with httpx.AsyncClient(
            base_url=upstream_url.rstrip("/"),
            headers=headers,
            timeout=httpx.Timeout(10.0, read=upstream_timeout),
            transport=transport,
        ) as client:
            app.state.client = client
            yield

    app = FastAPI(title="Unmimic", lifespan=lifespan)

    @app.get("/v1/models")
    async def list_models(request: Request) -> Response:
        try:
            upstream = await request.app.state.client.get("/models")
        except httpx.HTTPError as exc:
            return upstream_unreachable(exc)
        return passthrough(upstream)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        try:
            body = await request.json()
        except ValueError:
            body = None
        if not isinstance(body, dict):
            message = "request body must be a JSON object"
            return error_response(400, message, "invalid_request_error")

        ctx = Context(account_id=account_id_from(request), model=str(body.get("model", "")))
        for guard in inputs:
            verdict = guard.check(body, ctx)
            if verdict.action != "allow":
                log.warning(
                    "request %s: account=%s model=%s guard=%s score=%.2f %s",
                    "blocked" if verdict.action == "block" else "flagged",
                    ctx.account_id,
                    ctx.model,
                    guard.name,
                    verdict.score,
                    verdict.reason,
                )
            if verdict.action == "block":
                # Deliberately vague: telling attackers which guard fired helps them tune.
                return error_response(403, "request blocked by Unmimic", "unmimic_blocked")
            if verdict.action == "flag":
                flag = {"guard": guard.name, "score": verdict.score, "reason": verdict.reason}
                ctx.metadata.setdefault("flags", []).append(flag)

        client = request.app.state.client
        fmt = known_formats.get(ctx.model, DEFAULT_FORMAT)
        ctx.metadata["format"] = fmt  # output guards need it to separate answer from reasoning
        if body.get("stream"):
            return await stream_completion(client, body, ctx, fmt, outputs)
        return await complete(client, body, ctx, fmt, outputs)

    return app


async def complete(
    client: httpx.AsyncClient,
    body: dict[str, Any],
    ctx: Context,
    fmt: ModelFormat,
    output_guards: Sequence[OutputGuard],
) -> Response:
    try:
        upstream = await client.post("/chat/completions", json=body)
    except httpx.HTTPError as exc:
        return upstream_unreachable(exc)
    if upstream.status_code != 200 or not output_guards:
        return passthrough(upstream)

    data = upstream.json()
    reasoning = "\n".join(
        split_message(choice.get("message") or {}, fmt).reasoning
        for choice in data.get("choices", [])
    )
    for guard in output_guards:
        data = guard.transform(data, reasoning, ctx)
    return JSONResponse(data)


async def stream_completion(
    client: httpx.AsyncClient,
    body: dict[str, Any],
    ctx: Context,
    fmt: ModelFormat,
    output_guards: Sequence[OutputGuard],
) -> Response:
    try:
        upstream = await client.send(
            client.build_request("POST", "/chat/completions", json=body), stream=True
        )
    except httpx.HTTPError as exc:
        return upstream_unreachable(exc)
    if upstream.status_code != 200:
        await upstream.aread()
        await upstream.aclose()
        return passthrough(upstream)

    ctx.metadata["stream"] = True
    chunks: AsyncIterator[bytes] | AsyncIterator[str] = (
        guarded_events(upstream, ctx, fmt, output_guards)
        if output_guards
        else upstream.aiter_bytes()
    )
    return StreamingResponse(
        close_after(chunks, upstream),
        media_type=upstream.headers.get("content-type", "text/event-stream"),
    )


async def guarded_events(
    upstream: httpx.Response,
    ctx: Context,
    fmt: ModelFormat,
    output_guards: Sequence[OutputGuard],
) -> AsyncIterator[str]:
    """Run output guards on each server-sent event.

    Each guard sees one chunk plus all reasoning streamed so far, and may rewrite the chunk.
    """
    choices: dict[int, StreamReasoning] = {}
    async for line in upstream.aiter_lines():
        if not line.startswith("data:"):
            if line:  # comments and "event:" fields pass through
                yield line + "\n"
            continue
        payload = line.removeprefix("data:").strip()
        if payload == "[DONE]":
            yield "data: [DONE]\n\n"
            continue
        chunk = json.loads(payload)
        for choice in chunk.get("choices", []):
            tracker = choices.setdefault(choice.get("index", 0), StreamReasoning(fmt))
            tracker.feed(choice.get("delta") or {})
        reasoning = "\n".join(tracker.reasoning for tracker in choices.values())
        for guard in output_guards:
            chunk = guard.transform(chunk, reasoning, ctx)
        yield f"data: {json.dumps(chunk)}\n\n"


async def close_after(
    chunks: AsyncIterator[bytes] | AsyncIterator[str], upstream: httpx.Response
) -> AsyncIterator[bytes | str]:
    """Forward chunks, then close the upstream stream even if the client disconnects."""
    try:
        async for chunk in chunks:
            yield chunk
    finally:
        await upstream.aclose()

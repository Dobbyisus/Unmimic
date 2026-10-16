"""OpenAI-compatible proxy.

Every request passes the input guards, goes to the upstream model, and every response
passes the output guards on its way back. With no output guards configured, responses
are forwarded unchanged (streams byte for byte).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import AsyncIterator, Iterable, Sequence
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from unmimic.plugins import Context, InputGuard, OutputGuard

ANONYMOUS = "anonymous"
UPSTREAM_TIMEOUT = httpx.Timeout(10.0, read=600.0)


def account_id_from(request: Request) -> str:
    """Derive a stable account id from the caller's API key without keeping the key."""
    scheme, _, key = request.headers.get("authorization", "").partition(" ")
    key = key.strip()
    if scheme.lower() != "bearer" or not key:
        return ANONYMOUS
    return "acct_" + hashlib.sha256(key.encode()).hexdigest()[:16]


def extract_reasoning(message: dict[str, Any]) -> str:
    """Reasoning sent in a separate API field (vLLM ``reasoning_content``, Ollama ``reasoning``).

    Models that put reasoning inline in the text (``<think>`` tags) yield "" here until the
    format probe exists.
    """
    return message.get("reasoning_content") or message.get("reasoning") or ""


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
    transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    """Build the proxy. ``transport`` replaces the network, for tests."""
    inputs = list(input_guards)
    outputs = list(output_guards)
    headers = {"authorization": f"Bearer {upstream_api_key}"} if upstream_api_key else {}

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with httpx.AsyncClient(
            base_url=upstream_url.rstrip("/"),
            headers=headers,
            timeout=UPSTREAM_TIMEOUT,
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
            if verdict.action == "block":
                # Deliberately vague: telling attackers which guard fired helps them tune.
                return error_response(403, "request blocked by Unmimic", "unmimic_blocked")
            if verdict.action == "flag":
                flag = {"guard": guard.name, "score": verdict.score, "reason": verdict.reason}
                ctx.metadata.setdefault("flags", []).append(flag)

        client = request.app.state.client
        if body.get("stream"):
            return await stream_completion(client, body, ctx, outputs)
        return await complete(client, body, ctx, outputs)

    return app


async def complete(
    client: httpx.AsyncClient,
    body: dict[str, Any],
    ctx: Context,
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
        extract_reasoning(choice.get("message") or {}) for choice in data.get("choices", [])
    )
    for guard in output_guards:
        data = guard.transform(data, reasoning, ctx)
    return JSONResponse(data)


async def stream_completion(
    client: httpx.AsyncClient,
    body: dict[str, Any],
    ctx: Context,
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
        guarded_events(upstream, ctx, output_guards) if output_guards else upstream.aiter_bytes()
    )
    return StreamingResponse(
        close_after(chunks, upstream),
        media_type=upstream.headers.get("content-type", "text/event-stream"),
    )


async def guarded_events(
    upstream: httpx.Response, ctx: Context, output_guards: Sequence[OutputGuard]
) -> AsyncIterator[str]:
    """Run output guards on each server-sent event.

    Each guard sees one chunk plus all reasoning streamed so far, and may rewrite the chunk.
    """
    reasoning = ""
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
            reasoning += extract_reasoning(choice.get("delta") or {})
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

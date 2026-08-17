"""Local stdlib-only HTTP server for the RepoTriage Agent feature.

Binds 127.0.0.1 only. No web framework — http.server + ThreadingHTTPServer.
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

from llm_ladder import jobs
from llm_ladder.config import ChainConfig, TierConfig, default_chains_path, load_chains
from llm_ladder.digest import DigestOptions, run_digest
from llm_ladder.github_client import fetch_open_issues
from llm_ladder.model_config import ModelConfig, load_config, save_config
from llm_ladder.nl_router import route_command
from llm_ladder.ollama_manager import (
    CURATED_MODELS,
    is_model_pulled,
    list_installed_models,
    pull_model_job,
)
from llm_ladder.providers import APIKeyProvider, OllamaProvider
from llm_ladder.triage import triage_issue

DEFAULT_PORT = 8420
PORT = int(os.environ.get("LLM_LADDER_PORT", DEFAULT_PORT))
WEB_DIR = (Path(__file__).resolve().parent.parent / "web").resolve()

_ALLOWED_HOSTS = {f"127.0.0.1:{PORT}", f"localhost:{PORT}"}
_ALLOWED_ORIGINS = {f"http://127.0.0.1:{PORT}", f"http://localhost:{PORT}"}
_MAX_BODY_BYTES = 1 << 20  # 1 MiB, generous for a JSON command/model-config body

_STATIC_ROUTES = {
    "/": "index.html",
    "/index.html": "index.html",
    "/agent.html": "agent.html",
    "/digest.html": "digest.html",
    "/stats.html": "stats.html",
    "/benchmark.html": "benchmark.html",
}


class _BadRequestError(Exception):
    """Raised for client-side request problems that should map to HTTP 400."""


def _resolve_chain() -> tuple[str, ChainConfig, object | None]:
    """Return (chain_name, chain_config, provider) for the current ModelConfig.

    provider is None for 'default'/'local' modes (caller relies on the
    cascade / OllamaProvider); an APIKeyProvider for 'api_key' mode.
    """
    cfg = load_config()
    if cfg.mode == "api_key":
        provider = APIKeyProvider(
            base_url=cfg.base_url or "",
            api_key=os.environ.get("LLM_LADDER_API_KEY", ""),
            api_style=cfg.api_style,
        )
        chain_config = ChainConfig(name="api_key", tiers=[TierConfig(model=cfg.model, samples=1, threshold=0.0)])
        return "api_key", chain_config, provider
    if cfg.mode == "local":
        chain_config = ChainConfig(name="local", tiers=[TierConfig(model=cfg.model, samples=1, threshold=0.0)])
        return "local", chain_config, None
    chains = load_chains(default_chains_path())
    return "default", chains["default"], None


def _triage_job(repo: str, chain_name: str, chain_config: ChainConfig, provider) -> callable:
    def _run(report_progress):
        issues = fetch_open_issues(repo, token=os.environ.get("GITHUB_TOKEN"))
        titles = [f"#{issue.number}: {issue.title}" for issue in issues]
        results = []
        for issue in issues:
            result = triage_issue(issue, titles, chain_name, chain_config, provider=provider)
            results.append(dataclasses.asdict(result))
            report_progress({"completed": len(results), "total": len(issues), "results": results})
        return {"repo": repo, "results": results}

    return _run


def _digest_job(opts: DigestOptions) -> callable:
    def _run(report_progress):
        result = run_digest(opts, report_progress=report_progress)
        return dataclasses.asdict(result)

    return _run


class Handler(BaseHTTPRequestHandler):
    server_version = "llm-ladder-agent/0.1"

    # -- trust-boundary check -------------------------------------------------

    def _origin_allowed(self) -> bool:
        host = self.headers.get("Host", "")
        if host not in _ALLOWED_HOSTS:
            return False
        origin = self.headers.get("Origin")
        if origin is not None and origin not in _ALLOWED_ORIGINS:
            return False
        return True

    # -- helpers ----------------------------------------------------------

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> dict:
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            raise _BadRequestError("missing Content-Length header")
        try:
            length = int(raw_length)
        except (TypeError, ValueError):
            raise _BadRequestError("malformed Content-Length header")
        if length < 0:
            raise _BadRequestError("negative Content-Length header")
        if length > _MAX_BODY_BYTES:
            raise _BadRequestError(f"request body exceeds {_MAX_BODY_BYTES} bytes")
        raw = self.rfile.read(length) if length else b""
        return json.loads(raw or b"{}")

    def _serve_static(self, url_path: str) -> None:
        rel = _STATIC_ROUTES.get(url_path)
        if rel is None:
            rel = unquote(url_path).lstrip("/")
        target = (WEB_DIR / rel).resolve()
        if WEB_DIR not in target.parents and target != WEB_DIR:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        if not target.is_file():
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        content_type = "text/html" if target.suffix == ".html" else "application/octet-stream"
        body = target.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # -- dispatch -----------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler naming
        self._handle(self._route_get)

    def do_POST(self) -> None:  # noqa: N802
        self._handle(self._route_post)

    def _handle(self, route_fn) -> None:
        if not self._origin_allowed():
            self._send_json(HTTPStatus.FORBIDDEN, {"error": "forbidden host/origin"})
            return
        try:
            route_fn()
        except _BadRequestError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except json.JSONDecodeError:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "malformed JSON body"})
        except Exception as exc:  # noqa: BLE001 - never leak a traceback to the client
            traceback.print_exc(file=sys.stderr)
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

    def _route_get(self) -> None:
        path = urlsplit(self.path).path
        if path == "/api/models":
            cfg = load_config()
            self._send_json(
                HTTPStatus.OK,
                {
                    "config": dataclasses.asdict(cfg),
                    "curated": CURATED_MODELS,
                    "installed": list_installed_models(),
                },
            )
            return
        if path.startswith("/api/jobs/"):
            job_id = path[len("/api/jobs/"):]
            status = jobs.get_status(job_id)
            code = HTTPStatus.NOT_FOUND if status["error"] == f"unknown or expired job_id: {job_id}" else HTTPStatus.OK
            self._send_json(code, status)
            return
        if path.startswith("/api/"):
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        self._serve_static(path)

    def _route_post(self) -> None:
        path = urlsplit(self.path).path
        if path == "/api/command":
            body = self._read_json_body()
            text = str(body.get("text", ""))
            chain_name, chain_config, provider = _resolve_chain()
            tier1_model = chain_config.tiers[0].model
            chat_provider = provider or OllamaProvider()
            # Route via whichever backend is currently configured (local
            # Ollama or the API-key provider), not a hardcoded Ollama call.
            intent = route_command(text, model=tier1_model, chat_fn=chat_provider.chat)
            if intent.kind == "triage_repo":
                job_id = jobs.start_job(_triage_job(intent.repo, chain_name, chain_config, provider))
                self._send_json(HTTPStatus.ACCEPTED, {"kind": "triage_repo", "job_id": job_id})
                return
            answer = chat_provider.chat(text, model=tier1_model)
            self._send_json(HTTPStatus.OK, {"kind": "ask", "answer": answer})
            return
        if path == "/api/triage":
            body = self._read_json_body()
            repo = str(body.get("repo", ""))
            chain_name, chain_config, provider = _resolve_chain()
            job_id = jobs.start_job(_triage_job(repo, chain_name, chain_config, provider))
            self._send_json(HTTPStatus.ACCEPTED, {"job_id": job_id})
            return
        if path == "/api/digest":
            # No concurrency guard: two simultaneous --lens digests would
            # interleave free_loaded_models() calls and race each other's
            # memory checks, breaking the "never two large models at once"
            # guarantee. Trusted-single-user-machine assumption (same as
            # `serve` generally) — don't fire overlapping digest jobs.
            body = self._read_json_body()
            models = body.get("models")
            try:
                opts = DigestOptions(
                    repo=str(body.get("path", ".")),
                    releases=int(body.get("releases", 3)),
                    range_=body.get("range"),
                    file_path=body.get("file"),
                    lens=bool(body.get("lens", False)),
                    models=[str(m) for m in models] if models else None,
                    chain=str(body.get("chain", "default")),
                    since_last=bool(body.get("since_last", False)),
                )
            except (TypeError, ValueError) as exc:
                raise _BadRequestError(f"invalid digest options: {exc}")
            # DigestError from bad option combos or git failures surfaces via
            # job polling, same as any other _triage_job-style failure — not
            # raised synchronously here, since run_digest only runs once the
            # background thread starts.
            job_id = jobs.start_job(_digest_job(opts))
            self._send_json(HTTPStatus.ACCEPTED, {"job_id": job_id})
            return
        if path == "/api/models":
            body = self._read_json_body()
            api_key = body.pop("api_key", None)
            api_style = str(body.get("api_style", "openai"))
            cfg = ModelConfig(
                mode=str(body.get("mode", "")),
                model=str(body.get("model", "")),
                base_url=body.get("base_url"),
                api_style=api_style if api_style in ("openai", "anthropic") else "openai",
            )
            save_config(cfg)
            if api_key is not None:
                os.environ["LLM_LADDER_API_KEY"] = str(api_key)
            response = {"ok": True}
            if cfg.mode in ("default", "local"):
                # Report whether the saved model tag is already pulled so
                # the client can skip a redundant download.
                response["pulled"] = is_model_pulled(cfg.model or CURATED_MODELS[0])
            self._send_json(HTTPStatus.OK, response)
            return
        if path == "/api/models/pull":
            body = self._read_json_body()
            tag = str(body.get("tag", ""))
            job_id = pull_model_job(tag)
            self._send_json(HTTPStatus.OK, {"job_id": job_id})
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def log_message(self, fmt: str, *args) -> None:  # quieter default logging
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"Serving on http://127.0.0.1:{PORT}/agent.html")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

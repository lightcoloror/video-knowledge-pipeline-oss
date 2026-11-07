from __future__ import annotations

import argparse
import ipaddress
import json
import os
import secrets
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from .consented_model_batch import list_consented_model_batches
from .model_api_settings import (
    MODEL_TASKS,
    PROVIDER_PRESETS,
    install_model_api_onboarding_bundle,
    replace_model_api_route_configuration,
    delete_model_api_profile,
    public_model_api_settings_status,
    upsert_model_api_profile,
    validate_model_api_profile,
)


from .model_provider_probe import probe_model_api_onboarding_bundle
from .model_api_settings import (
    load_model_api_settings_ui_config,
    project_root as resolve_project_root,
)
from .model_route_settings import TASK_CAPABILITIES
from .model_screening_lab import simulate_offline_gateway_contract
MAX_REQUEST_BYTES = 128 * 1024
PROFILE_FIELDS = (
    "id",
    "name",
    "provider",
    "litellm_provider",
    "provider_options",
    "adapter_backend",
    "location",
    "capabilities",
    "base_url",
    "model",
    "timeout_seconds",
    "rpm",
    "tpm",
    "max_parallel_requests",
    "enabled",
)
TASK_LABELS = {
    "asr": "在线 ASR",
    "ocr": "在线 OCR",
    "document_visual": "PPT / 文档视觉",
    "semantic_frame": "单帧语义",
    "temporal_sequence": "多帧时序",
    "video_segment": "视频片段视觉",
    "text_llm": "通用文本模型",
    "summary_rewrite": "智能摘要改写",
    "transcript_correction": "转录纠错",
}


def build_server(
    *,
    host: str = "127.0.0.1",
    port: int = 8767,
    settings_path: str | Path | None = None,
    secrets_path: str | Path | None = None,
    csrf_token: str | None = None,
    project_root_path: str | Path | None = None,
) -> ThreadingHTTPServer:
    _require_loopback_host(host)
    token = str(csrf_token or secrets.token_urlsafe(32))

    class Handler(ModelApiSettingsHandler):
        pass

    Handler.settings_path = settings_path
    Handler.secrets_path = secrets_path
    Handler.csrf_token = token
    Handler.batch_project_root = Path(
        project_root_path if project_root_path is not None else resolve_project_root()
    ).expanduser().resolve()
    server = ThreadingHTTPServer((host, int(port)), Handler)
    server.daemon_threads = True
    return server


class ModelApiSettingsHandler(BaseHTTPRequestHandler):
    server_version = "VKPModelApiSettings/1.0"
    settings_path: str | Path | None = None
    secrets_path: str | Path | None = None
    csrf_token: str = ""
    batch_project_root: Path = resolve_project_root()

    def do_GET(self) -> None:  # noqa: N802
        if not self._host_allowed():
            return self._json_error(HTTPStatus.MISDIRECTED_REQUEST, "invalid Host header")
        path = urlsplit(self.path).path
        if path in {"/", "/index.html"}:
            return self._html_response(_render_html(self.csrf_token))
        if path == "/healthz":
            return self._json_response({"ok": True, "service": "vkp-model-api-settings", "loopback_only": True})
        if path == "/api/settings":
            return self._json_response(self._public_status())
        if path == "/api/model-batches":
            return self._json_response(
                list_consented_model_batches(self.batch_project_root)
            )
        return self._json_error(HTTPStatus.NOT_FOUND, "not found")

    def do_POST(self) -> None:  # noqa: N802
        if not self._prepare_mutation():
            return
        path = urlsplit(self.path).path
        if path not in {"/api/validate", "/api/screening/simulate"} and not path.startswith("/api/onboarding-catalog/"):
            return self._json_error(HTTPStatus.NOT_FOUND, "not found")
        try:
            payload = self._read_json_body()
            if path.startswith("/api/onboarding-catalog/"):
                provider_id = unquote(path.removeprefix("/api/onboarding-catalog/"))
                unexpected = sorted(set(payload) - {"execute"})
                if unexpected:
                    raise ValueError(
                        "catalog probe payload only accepts execute; unexpected fields: "
                        + ", ".join(unexpected)
                    )
                if payload.get("execute") is not True:
                    raise ValueError("catalog probe requires execute=true from an explicit operator action")
                probe = probe_model_api_onboarding_bundle(
                    provider_id,
                    execute=True,
                    settings_path=self.settings_path,
                    secrets_path=self.secrets_path,
                )
                return self._json_response({"ok": True, "probe": probe})
            if path == "/api/screening/simulate":
                simulation = simulate_offline_gateway_contract(payload.get("task"), payload.get("scenario"))
                return self._json_response({"ok": True, "simulation": simulation})
            profile = _profile_payload(payload)
            validated = validate_model_api_profile(profile, payload.get("tasks"))
            validated["profile"]["api_key_configured"] = bool(payload.get("api_key"))
            return self._json_response({"ok": True, "validated": validated})
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            return self._json_error(HTTPStatus.BAD_REQUEST, str(exc))

    def do_PUT(self) -> None:  # noqa: N802
        if not self._prepare_mutation():
            return
        path = urlsplit(self.path).path
        try:
            payload = self._read_json_body()
            if path == "/api/profile":
                profile = _profile_payload(payload)
                result = upsert_model_api_profile(
                    profile,
                    tasks=payload.get("tasks"),
                    api_key=str(payload.get("api_key") or ""),
                    remove_api_key=bool(payload.get("remove_api_key")),
                    settings_path=self.settings_path,
                    secrets_path=self.secrets_path,
                )
            elif path.startswith("/api/onboarding/"):
                provider_id = unquote(path.removeprefix("/api/onboarding/"))
                unexpected = sorted(set(payload) - {"api_key"})
                if unexpected:
                    raise ValueError(
                        "onboarding payload only accepts api_key; unexpected fields: "
                        + ", ".join(unexpected)
                    )
                result = install_model_api_onboarding_bundle(
                    provider_id,
                    api_key=str(payload.get("api_key") or ""),
                    settings_path=self.settings_path,
                    secrets_path=self.secrets_path,
                )
            elif path == "/api/routes":
                route_pools = payload.get("route_pools")
                route_bindings = payload.get("route_bindings")
                if not isinstance(route_pools, list) or not isinstance(route_bindings, dict):
                    raise ValueError("route_pools must be a list and route_bindings must be an object")
                result = replace_model_api_route_configuration(
                    route_pools,
                    route_bindings,
                    settings_path=self.settings_path,
                    secrets_path=self.secrets_path,
                )
            else:
                return self._json_error(HTTPStatus.NOT_FOUND, "not found")
            return self._json_response({"ok": True, "settings": result})
        except (ValueError, TypeError, json.JSONDecodeError, RuntimeError, OSError) as exc:
            return self._json_error(HTTPStatus.BAD_REQUEST, str(exc))
    def do_DELETE(self) -> None:  # noqa: N802
        if not self._prepare_mutation():
            return
        path = urlsplit(self.path).path
        prefix = "/api/profile/"
        if not path.startswith(prefix):
            return self._json_error(HTTPStatus.NOT_FOUND, "not found")
        profile_id = unquote(path[len(prefix) :])
        try:
            result = delete_model_api_profile(
                profile_id,
                settings_path=self.settings_path,
                secrets_path=self.secrets_path,
            )
            return self._json_response({"ok": True, "settings": result})
        except (ValueError, RuntimeError, OSError) as exc:
            return self._json_error(HTTPStatus.BAD_REQUEST, str(exc))

    def log_message(self, format: str, *args: Any) -> None:
        safe_args = tuple(str(value).replace("\r", " ").replace("\n", " ") for value in args)
        super().log_message(format, *safe_args)

    def _public_status(self) -> dict[str, Any]:
        return public_model_api_settings_status(self.settings_path, self.secrets_path)

    def _prepare_mutation(self) -> bool:
        if not self._host_allowed():
            self._json_error(HTTPStatus.MISDIRECTED_REQUEST, "invalid Host header")
            return False
        if self.headers.get("X-VKP-Settings-Token", "") != self.csrf_token:
            self._json_error(HTTPStatus.FORBIDDEN, "invalid settings token")
            return False
        content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        if content_type != "application/json":
            self._json_error(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "Content-Type must be application/json")
            return False
        return True

    def _host_allowed(self) -> bool:
        raw = str(self.headers.get("Host") or "").strip()
        if not raw:
            return False
        host = raw
        if raw.startswith("[") and "]" in raw:
            host = raw[1 : raw.index("]")]
        elif raw.count(":") == 1:
            host = raw.rsplit(":", 1)[0]
        return _is_loopback_host(host)

    def _read_json_body(self) -> dict[str, Any]:
        raw_length = str(self.headers.get("Content-Length") or "0")
        length = int(raw_length)
        if length < 1 or length > MAX_REQUEST_BYTES:
            raise ValueError(f"request body must be between 1 and {MAX_REQUEST_BYTES} bytes")
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload

    def _json_response(self, value: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(int(status))
        self._security_headers("application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _json_error(self, status: HTTPStatus, message: str) -> None:
        self._json_response({"ok": False, "error": str(message)}, status)

    def _html_response(self, value: str) -> None:
        payload = value.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self._security_headers("text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _security_headers(self, content_type: str) -> None:
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; style-src 'unsafe-inline'; script-src 'self' 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'",
        )


def _profile_payload(payload: dict[str, Any]) -> dict[str, Any]:
    raw = payload.get("profile")
    if not isinstance(raw, dict):
        raise ValueError("profile must be an object")
    return {key: raw[key] for key in PROFILE_FIELDS if key in raw}


def _render_html(csrf_token: str) -> str:
    providers_json = json.dumps(list(PROVIDER_PRESETS), ensure_ascii=False).replace("</", "<\\/")
    tasks_json = json.dumps(
        [
            {"id": task, "label": TASK_LABELS[task], "capability": TASK_CAPABILITIES[task]}
            for task in MODEL_TASKS
        ],
        ensure_ascii=False,
    ).replace("</", "<\\/")
    token_json = json.dumps(csrf_token).replace("</", "<\\/")
    template = r'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>VKP 混合模型控制台</title>
  <style>
    :root{color-scheme:dark;--ink:#eaf1f3;--muted:#93a5ab;--bg:#071014;--panel:#0c181d;--panel2:#101f25;--line:#294047;--cyan:#4dd5df;--amber:#ffb84d;--ok:#5bd19b;--bad:#ff756f;--shadow:rgba(0,0,0,.35)}
    *{box-sizing:border-box}body{margin:0;background:linear-gradient(rgba(77,213,223,.035) 1px,transparent 1px),linear-gradient(90deg,rgba(77,213,223,.035) 1px,transparent 1px),radial-gradient(circle at 75% 0,#15313a 0,transparent 38%),var(--bg);background-size:28px 28px,28px 28px,auto;color:var(--ink);font:14px/1.55 "Microsoft YaHei UI","Noto Sans CJK SC",sans-serif}
    header{border-bottom:1px solid var(--line);padding:30px max(22px,calc((100vw - 1320px)/2));position:relative;overflow:hidden}header:after{content:"ROUTE / CONSENT / RUNTIME";position:absolute;right:max(22px,calc((100vw - 1320px)/2));top:20px;color:rgba(77,213,223,.13);font:700 28px/1 Bahnschrift,Consolas,monospace;letter-spacing:.12em}h1,h2,h3{font-family:Bahnschrift,"Microsoft YaHei UI",sans-serif;letter-spacing:.02em}header h1{margin:0;color:#fff;font-size:30px}header p{margin:6px 0 0;color:var(--muted);max-width:760px}
    main{max-width:1320px;margin:auto;padding:20px 22px 48px}.notice,.card{background:linear-gradient(145deg,rgba(16,31,37,.96),rgba(8,20,25,.97));border:1px solid var(--line);box-shadow:0 14px 34px var(--shadow)}.notice{padding:13px 16px;margin-bottom:16px;border-left:4px solid var(--amber)}.notice strong{color:var(--amber)}
    .layout{display:grid;grid-template-columns:330px minmax(0,1fr);gap:16px}.card{padding:18px;border-radius:3px}.card h2{margin:0 0 13px;font-size:18px;color:#fff}.toolbar{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px}button{border:1px solid #38545d;background:#13242a;color:var(--ink);padding:9px 13px;font-weight:700;cursor:pointer;letter-spacing:.02em}button:hover{border-color:var(--cyan);color:#fff}button.primary{background:var(--cyan);border-color:var(--cyan);color:#042126}button.danger{color:var(--bad)}button:disabled{opacity:.45;cursor:not-allowed}
    .profile{border:1px solid var(--line);padding:11px;margin:8px 0;cursor:pointer;background:rgba(4,13,16,.5)}.profile:hover,.profile.active{border-color:var(--cyan);box-shadow:inset 3px 0 var(--cyan)}.profile-title{display:flex;justify-content:space-between;gap:8px;font-weight:750}.tag{display:inline-block;border:1px solid #31515a;padding:1px 7px;color:#a9d8de;font-size:11px;margin:5px 4px 0 0;text-transform:uppercase}.tag.key{border-color:#356950;color:var(--ok)}.tag.no-key{border-color:#755b32;color:var(--amber)}
    .grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}label{display:block;color:#a8bac0;font-size:12px;font-weight:750;margin:3px 0 5px;text-transform:uppercase;letter-spacing:.04em}input,select{width:100%;border:1px solid #35505a;background:#071216;color:var(--ink);padding:10px}input:focus,select:focus{outline:1px solid var(--cyan);border-color:var(--cyan)}.full{grid-column:1/-1}.checks{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:7px}.check{display:flex;align-items:center;gap:7px;border:1px solid var(--line);padding:8px;background:#0a1519}.check input{width:auto}.check label{margin:0;color:var(--ink);font-weight:500;text-transform:none;letter-spacing:0}
    .secret-row{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:9px;align-items:end}.muted{color:var(--muted)}code{color:#bceef1;background:#071216;padding:2px 5px;word-break:break-all}.status{min-height:24px;margin-top:10px;font-weight:700}.status.ok{color:var(--ok)}.status.error{color:var(--bad)}.paths{font-size:11px;margin-top:9px}.empty{padding:22px 8px;text-align:center;color:var(--muted)}
    .routes{margin-top:16px}.route-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}.route-row,.pool-row,.route-state{border:1px solid var(--line);background:#091418;padding:11px}.route-row h3,.pool-row h3{margin:0 0 8px;font-size:14px}.route-controls{display:grid;grid-template-columns:110px 1fr 1fr;gap:8px}.pool-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;margin:12px 0}.pool-meta{color:var(--muted);font-size:12px;margin-bottom:7px}.route-state{font-size:12px}.route-state strong{color:var(--cyan)}.route-state .warn{color:var(--amber)}.revision{font-family:Consolas,monospace;color:#bfd2d6;word-break:break-all}.section-note{color:var(--muted);margin:-6px 0 12px}
    .onboarding-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}.onboarding-row{border:1px solid var(--line);background:#091418;padding:13px;display:flex;flex-direction:column;gap:8px}.onboarding-row h3{margin:0;font-size:15px}.onboarding-links{display:flex;gap:10px;flex-wrap:wrap}.onboarding-links a{color:var(--cyan);font-size:12px}.readiness{font-size:12px;color:#c6d6da}.readiness .warn{color:var(--amber)}
    .onboarding-model{border-left:2px solid #31515a;padding-left:8px;font-size:12px}.onboarding-key{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:8px}.onboarding-key input{min-width:0}
    .screening-controls{display:grid;grid-template-columns:1fr 1fr auto;gap:10px;align-items:end}.screening-output{white-space:pre-wrap;min-height:130px;border:1px solid var(--line);background:#061014;color:#bceef1;padding:12px;overflow:auto}.criteria-grid{display:flex;gap:8px;flex-wrap:wrap;margin:10px 0}.advanced{margin:16px 0}.advanced>summary{cursor:pointer;color:var(--cyan);font-weight:700;padding:4px 0 14px}.advanced[open]>summary{border-bottom:1px solid var(--line);margin-bottom:16px}
    @media(max-width:930px){.screening-controls{grid-template-columns:1fr}}
    @media(max-width:930px){.layout,.route-grid,.pool-grid,.onboarding-grid{grid-template-columns:1fr}.checks{grid-template-columns:repeat(2,minmax(0,1fr))}header:after{display:none}}@media(max-width:580px){main{padding:12px}.grid{grid-template-columns:1fr}.checks,.route-controls{grid-template-columns:1fr}.secret-row{grid-template-columns:1fr}}
  </style>
</head>
<body>
<header><h1>VKP 混合模型控制台</h1><p>把文本、视觉、ASR 与 OCR 映射到隔离的本地池或远程池；Agent 能力不参与路由决策。</p></header>
<main>
  <section class="notice"><strong>保存配置不等于授权外发。</strong> 远程任务仍需匹配 route revision 的 consent 与 Broker allowlist。API Key 只使用 Windows DPAPI 持久化，页面和接口不会回显。</section>
  <section class="card routes">
    <h2>快速接入（推荐：只填写 API Key）</h2>
    <p class="section-note">选择供应商并填写一次 Key；VKP 自动保存已审核的 Base URL、协议、精确模型、能力、Provider 参数与建议任务。不会联网、改默认路由、创建 consent 或授权外发。</p>
    <div id="freeOnboarding" class="onboarding-grid"></div>
  </section>
  <details class="advanced">
    <summary>高级配置：手工编辑单个 Provider、URL、模型与 JSON 参数</summary>
    <div class="layout">
    <aside class="card"><div class="toolbar"><button class="primary" id="newBtn">新建供应商</button><button id="refreshBtn">刷新</button></div><div id="profiles"></div><div class="paths" id="paths"></div></aside>
    <section class="card">
      <h2 id="formTitle">新建供应商</h2>
      <form id="profileForm" autocomplete="off">
        <input id="profileId" type="hidden">
        <div class="grid">
          <div><label for="name">配置名称</label><input id="name" maxlength="100" required placeholder="例如：Ark 视觉主力"></div>
          <div><label for="provider">供应商接口</label><select id="provider"></select></div>
          <div><label for="litellmProvider">LiteLLM provider prefix</label><input id="litellmProvider" maxlength="64" placeholder="例如 anthropic / groq / openai"></div>
          <div><label for="location">执行位置</label><select id="location"><option value="remote">远程 / 需 consent</option><option value="local">本地 loopback</option></select></div>
          <div><label for="adapter">运行模式</label><select id="adapter"><option value="proxy">LiteLLM Proxy（新配置默认）</option><option value="legacy">Legacy（显式兼容）</option><option value="litellm">Embedded LiteLLM SDK</option><option value="builtin">VKP 内置适配器</option><option value="auto">旧自动模式</option></select></div>
          <div><label for="timeout">超时（秒）</label><input id="timeout" type="number" min="1" max="3600" value="120"></div>
          <div><label for="rpm">Provider RPM</label><input id="rpm" type="number" min="1" max="1000000" placeholder="留空：不猜测"></div>
          <div><label for="tpm">Provider TPM</label><input id="tpm" type="number" min="1" max="100000000" placeholder="留空：不猜测"></div>
          <div><label for="maxParallelRequests">单 Deployment 最大并发</label><input id="maxParallelRequests" type="number" min="1" max="128" placeholder="留空：使用 LiteLLM / Provider 契约"></div>
          <div class="muted">三项配额直接交给 LiteLLM Router；请填写供应商控制台显示的真实额度。留空不会由 VKP 自行猜测。</div>
          <div class="full"><label for="baseUrl">Provider Base URL</label><input id="baseUrl" placeholder="https://api.example.com/v1 或 http://127.0.0.1:8000/v1"></div>
          <div class="full"><label for="model">模型 / Deployment</label><input id="model" maxlength="300" placeholder="模型或 LiteLLM model id"></div>
          <div class="full"><label for="providerOptions">Provider 非密钥参数（JSON）</label><input id="providerOptions" placeholder='例如 {"api_version":"2024-08-01-preview"}'><div class="muted" id="providerOptionsHint"></div></div>
          <div class="full muted" id="authHint">认证要求将在选择供应商后显示</div>
          <div class="full"><label>能力声明</label><div class="checks" id="capabilities"></div></div>
          <div class="full"><label>分配任务</label><div class="checks" id="tasks"></div></div>
          <div class="full"><label for="apiKey">API Key</label><div class="secret-row"><input id="apiKey" type="password" autocomplete="new-password" placeholder="留空则保留现有凭据"><button type="button" id="toggleKey">显示/隐藏</button></div><div class="muted" id="keyState">尚未保存凭据</div></div>
          <div class="full check"><input id="enabled" type="checkbox" checked><label for="enabled">启用此 profile</label></div>
          <div class="full check"><input id="removeKey" type="checkbox"><label for="removeKey">保存时删除已有 API Key</label></div>
        </div>
        <div class="toolbar" style="margin-top:15px"><button class="primary" type="submit">保存到本机</button><button type="button" id="validateBtn">仅验证，不保存</button><button class="danger" type="button" id="deleteBtn" disabled>删除 profile</button></div>
        <div id="status" class="status"></div>
      </form>
    </section>
    </div>
  </details>
  <section class="card routes">
    <h2>Offline Model Screening Lab</h2>
    <p class="section-note">This loopback-only lab exercises response and recovery contracts. Simulated latency, cost, and success never count as model-quality evidence.</p>
    <div class="screening-controls">
      <div><label for="screeningTask">Capability</label><select id="screeningTask"><option value="text">Text</option><option value="vision">Vision</option><option value="asr">ASR</option><option value="ocr">OCR</option></select></div>
      <div><label for="screeningScenario">Scenario</label><select id="screeningScenario"></select></div>
      <button type="button" class="primary" id="runScreening">Run offline simulation</button>
    </div>
    <div id="screeningCriteria" class="criteria-grid"></div>
    <div id="screeningCandidates" class="route-grid"></div>
    <pre id="screeningResult" class="screening-output">No simulation has run.</pre>
    <div id="screeningStatus" class="status"></div>
  </section>
  <section class="card routes">
    <h2>路由池与任务默认位置</h2>
    <p class="section-note">池内 deployment 按从左到右的顺序交给 LiteLLM；本地池与远程池不能混合。修改池内容会生成新的 route revision，旧 consent 自动失配。</p>
    <div id="poolEditor" class="pool-grid"></div>
    <div id="routeEditor" class="route-grid"></div>
    <div class="toolbar" style="margin-top:14px"><button class="primary" id="saveRoutes">保存路由配置</button></div>
    <div id="routeStatus" class="status"></div>
  </section>
  <section class="card routes">
    <h2>在线模型批次运行状态</h2>
    <p class="section-note">只读展示持久化队列、429、5xx/超时和依赖阻断。RPM、TPM、单 deployment 并发及冷却由 LiteLLM Router 负责；批次层不自研动态限流。</p>
    <div class="toolbar"><button type="button" id="refreshBatches">刷新批次状态</button></div>
    <div id="modelBatches" class="route-grid"></div>
    <div id="batchStatus" class="status"></div>
  </section>
  <section class="card routes">
    <h2>本地视频结构与通用标签能力</h2>
    <p class="section-note">镜头切分、语义场景、剧情线和通用标签均保持本地；高光模型需显式安装与执行。整段视频理解当前禁用。</p>
    <div id="localMediaCapabilities" class="route-grid"></div>
  </section>
  <section class="card routes">
    <h2>MediaKit 远程媒体能力</h2>
    <p class="section-note">这些是独立的异步媒体服务，不是 LiteLLM 模型。每次执行仍需绑定文件哈希、调用/费用上限和明确 consent；满足后由官方 CLI 执行提供方托管的本地上传。</p>
    <div id="mediaCapabilities" class="route-grid"></div>
  </section>
</main>
<script>
const CSRF=__CSRF_TOKEN__; const PRESETS=__PROVIDERS__; const TASKS=__TASKS__;
const CAPS=[{id:"text",label:"文本"},{id:"vision",label:"视觉"},{id:"asr",label:"ASR"},{id:"ocr",label:"OCR"}];
let state={profiles:[],task_routes:{},route_pools:[],route_bindings:{},route_status:[],free_screening_onboarding:{entries:[]},media_capability_catalog:{capabilities:[]}}; let batchState={items:[]}; let activeId="";
const $=id=>document.getElementById(id); const esc=value=>String(value??"").replace(/[&<>\"]/g,ch=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[ch]));
function message(text,ok=true,target="status"){const el=$(target);el.textContent=text;el.className="status "+(ok?"ok":"error")}
async function request(url,options={}){const headers={...(options.headers||{})};if(options.body){headers["Content-Type"]="application/json";headers["X-VKP-Settings-Token"]=CSRF}const response=await fetch(url,{...options,headers,cache:"no-store"});const data=await response.json();if(!response.ok||data.ok===false)throw new Error(data.error||`HTTP ${response.status}`);return data}
function initOptions(){$("provider").innerHTML=PRESETS.map(row=>`<option value="${row.provider}">${esc(row.label)}</option>`).join("");$("tasks").innerHTML=TASKS.map(row=>`<div class="check"><input type="checkbox" id="task-${row.id}" value="${row.id}"><label for="task-${row.id}">${esc(row.label)}</label></div>`).join("");$("capabilities").innerHTML=CAPS.map(row=>`<div class="check"><input type="checkbox" id="cap-${row.id}" value="${row.id}"><label for="cap-${row.id}">${row.label}</label></div>`).join("")}
function profileTasks(id){return TASKS.map(x=>x.id).filter(task=>state.task_routes[task]===id)}
function capacityText(row){const values=[];if(row.rpm)values.push(`RPM ${row.rpm}`);if(row.tpm)values.push(`TPM ${row.tpm}`);if(row.max_parallel_requests)values.push(`并发 ${row.max_parallel_requests}`);return values.length?values.join(" · "):"Provider 配额未登记"}
function render(){const host=$("profiles");if(!state.profiles.length){host.innerHTML='<div class="empty">还没有供应商配置</div>'}else{host.innerHTML=state.profiles.map(row=>{const tasks=profileTasks(row.id);return `<div class="profile ${row.id===activeId?'active':''}" data-id="${row.id}"><div class="profile-title"><span>${esc(row.name)}</span><span>${row.enabled?'ON':'OFF'}</span></div><div class="muted">${esc(row.location)} · ${esc(row.provider)} · ${esc(row.model||'未指定模型')}</div><div class="muted">${esc(capacityText(row))}</div><div><span class="tag">${esc(row.adapter_backend)}</span>${(row.capabilities||[]).map(x=>`<span class="tag">${esc(x)}</span>`).join('')}${tasks.map(t=>`<span class="tag">${esc(TASKS.find(x=>x.id===t)?.label||t)}</span>`).join('')}<span class="tag ${row.api_key_configured?'key':'no-key'}">${row.api_key_configured?'KEY READY':'NO KEY'}</span></div></div>`}).join('');host.querySelectorAll('.profile').forEach(el=>el.addEventListener('click',()=>edit(el.dataset.id)))}$("paths").innerHTML=`配置：<code>${esc(state.settings_path||'')}</code><br>凭据：<code>${esc(state.secrets_path||'')}</code>`;renderRoutes();renderLocalMediaCapabilities();renderMediaCapabilities()}
function reset(){activeId="";$("profileForm").reset();$("profileId").value="";$("adapter").value="proxy";$("location").value="remote";$("timeout").value="120";$("rpm").value="";$("tpm").value="";$("maxParallelRequests").value="";$("enabled").checked=true;$("removeKey").checked=false;$("keyState").textContent="尚未保存凭据";$("formTitle").textContent="新建供应商";$("deleteBtn").disabled=true;TASKS.forEach(x=>$("task-"+x.id).checked=false);applyPreset();render();message("")}
function authHint(row,status={}){const external=row?.auth_mode==="external_environment";$("apiKey").disabled=external;$("removeKey").disabled=external;const required=row?.required_provider_options||[];const env=status.required_environment||(row?.environment_bindings||[]).filter(x=>x.required).map(x=>x.env);$("authHint").textContent=`认证：${row?.auth_mode||'api_key_dpapi'}；必填参数：${required.join(', ')||'无'}；环境变量：${env.join(', ')||'无'}；当前状态：${status.status||'尚未检查'}`}
function providerOptionsHint(row){const allowed=row?.allowed_provider_options||[];let example="{}";if(row?.provider==="siliconflow")example='{"enable_thinking":false}';else if(row?.provider==="volcengine_coding_plan"||row?.provider==="volcengine_ark")example='{"thinking_mode":"disabled"}';$("providerOptions").placeholder=example;$("providerOptionsHint").textContent=`允许字段：${allowed.join(', ')||'无'}。SiliconFlow 使用 enable_thinking；火山方舟使用 thinking_mode。`}
function edit(id){const row=state.profiles.find(x=>x.id===id);if(!row)return;activeId=id;$("profileId").value=row.id;$("name").value=row.name;$("provider").value=row.provider;$("litellmProvider").value=row.litellm_provider||"";$("litellmProvider").readOnly=row.provider!=="litellm_native";$("adapter").value=row.adapter_backend||"legacy";$("location").value=row.location||"remote";$("baseUrl").value=row.base_url||"";$("model").value=row.model||"";$("providerOptions").value=JSON.stringify(row.provider_options||{});$("timeout").value=row.timeout_seconds||120;$("rpm").value=row.rpm||"";$("tpm").value=row.tpm||"";$("maxParallelRequests").value=row.max_parallel_requests||"";$("enabled").checked=row.enabled!==false;$("apiKey").value="";$("removeKey").checked=false;$("keyState").textContent=row.api_key_configured?"已保存 DPAPI 加密凭据；留空将保留":"尚未保存凭据";const preset=PRESETS.find(x=>x.provider===row.provider)||row;authHint(preset,row.auth_status||{});providerOptionsHint(preset);$("formTitle").textContent="编辑供应商";$("deleteBtn").disabled=false;const tasks=new Set(profileTasks(id));TASKS.forEach(x=>$("task-"+x.id).checked=tasks.has(x.id));const caps=new Set(row.capabilities||[]);CAPS.forEach(x=>$("cap-"+x.id).checked=caps.has(x.id));render();message("")}
function applyPreset(){const row=PRESETS.find(x=>x.provider===$("provider").value);if(!row)return;$("litellmProvider").value=row.litellm_provider||"";$("litellmProvider").readOnly=row.provider!=="litellm_native";if(!$("baseUrl").value)$("baseUrl").value=row.default_base_url||"";if(!$("model").value)$("model").value=row.default_model||"";$("providerOptions").value=JSON.stringify(Object.fromEntries((row.required_provider_options||[]).map(key=>[key,""])));authHint(row);providerOptionsHint(row);const defaults=new Set(row.default_capabilities||[]);CAPS.forEach(x=>$("cap-"+x.id).checked=defaults.has(x.id));const base=$("baseUrl").value.toLowerCase();$("location").value=(base.includes("127.0.0.1")||base.includes("localhost")||base.includes("[::1]"))?"local":(row.default_location||"remote")}
function optionalInt(id){const value=$(id).value.trim();return value===""?undefined:Number(value)}
function payload(){return {profile:{id:$("profileId").value||undefined,name:$("name").value,provider:$("provider").value,litellm_provider:$("litellmProvider").value,provider_options:JSON.parse($("providerOptions").value||"{}"),adapter_backend:$("adapter").value,location:$("location").value,capabilities:CAPS.map(x=>x.id).filter(id=>$("cap-"+id).checked),base_url:$("baseUrl").value,model:$("model").value,timeout_seconds:Number($("timeout").value||120),rpm:optionalInt("rpm"),tpm:optionalInt("tpm"),max_parallel_requests:optionalInt("maxParallelRequests"),enabled:$("enabled").checked},tasks:TASKS.map(x=>x.id).filter(id=>$("task-"+id).checked),api_key:$("apiKey").value,remove_api_key:$("removeKey").checked}}
function poolOptions(location,capability,current){const type=location==="local"?"local_only":"remote_approved";return '<option value="">未配置</option>'+state.route_pools.filter(p=>p.location===type&&p.capability===capability).map(p=>`<option value="${esc(p.id)}" ${p.id===current?'selected':''}>${esc(p.name)} (${esc(p.id)})</option>`).join('')}
function renderRoutes(){const pools=$("poolEditor");pools.innerHTML=(state.route_pools||[]).map(pool=>`<div class="pool-row" data-pool="${esc(pool.id)}"><h3>${esc(pool.name)}</h3><div class="pool-meta">${esc(pool.location)} · ${esc(pool.capability)} · <span class="revision">${esc(pool.id)}</span></div><label>有序 deployments（逗号分隔 profile id）</label><input class="pool-deployments" value="${esc((pool.deployments||[]).join(', '))}"></div>`).join('')||'<div class="empty">保存 profile 并分配任务后会生成隔离池</div>';
 const host=$("routeEditor");host.innerHTML=TASKS.map(task=>{const b=state.route_bindings?.[task.id]||{default_location:"remote",local_pool_id:"",remote_pool_id:""};const statuses=(state.route_status||[]).filter(x=>x.task===task.id);const evidence=statuses.map(s=>`<div class="route-state"><strong>${esc(s.execution_location)}</strong> · health ${esc(s.health_status||s.status||'unknown')} · cost ${esc(s.estimated_cost||'unknown')} · consent ${s.consent_required?'required':'not required'} · allowlist <span class="${s.allowlist_status==='approved'||s.allowlist_status==='not_required'?'':'warn'}">${esc(s.allowlist_status||'unknown')}</span><br><span class="revision">rev ${esc((s.route_revision||'').slice(0,16))}…</span></div>`).join('');return `<div class="route-row" data-task="${task.id}"><h3>${esc(task.label)} <span class="tag">${esc(task.capability)}</span></h3><div class="route-controls"><div><label>默认位置</label><select class="route-default"><option value="local" ${b.default_location==='local'?'selected':''}>本地</option><option value="remote" ${b.default_location!=='local'?'selected':''}>远程</option></select></div><div><label>本地池</label><select class="route-local">${poolOptions('local',task.capability,b.local_pool_id)}</select></div><div><label>远程池</label><select class="route-remote">${poolOptions('remote',task.capability,b.remote_pool_id)}</select></div></div>${evidence}</div>`}).join('')}
function renderLocalMediaCapabilities(){const host=$("localMediaCapabilities");const rows=state.media_capability_catalog?.local_video_structure_capabilities||[];host.innerHTML=rows.map(row=>`<div class="pool-row"><h3>${esc(row.label)} <span class="tag">${esc(row.status)}</span></h3><div class="muted">${esc(row.task)} · ${esc(row.implementation)}${row.default_model?' · '+esc(row.default_model):''}</div><div class="route-state"><strong>${row.model_required?'optional local model':'local evidence pipeline'}</strong>${row.device_policy?' · '+esc(row.device_policy):''} · cloud ${row.cloud_allowed?'allowed':'disabled'} · candidate evidence</div></div>`).join('')||'<div class="empty">没有本地视频结构能力</div>'}
function renderMediaCapabilities(){const host=$("mediaCapabilities");const rows=state.media_capability_catalog?.capabilities||[];host.innerHTML=rows.map(row=>`<div class="pool-row"><h3>${esc(row.label)} <span class="tag">${esc(row.priority)}</span></h3><div class="muted">${esc(row.task)} · ${esc(row.provider_task)} · ${esc(row.protocol)}</div><div class="route-state"><strong>candidate only</strong> · consent required · allowlist <span class="warn">not approved</span> · cost ${esc(row.estimated_cost||'unknown')}<br><span class="revision">upload destinations: ${esc(row.upload_destinations_status||'not audited')}</span></div></div>`).join('')||'<div class="empty">没有已登记的媒体能力</div>'}
function renderBatches(){const host=$("modelBatches");const rows=batchState.items||[];host.innerHTML=rows.map(row=>{const s=row.summary||{};const issue=(s.rate_limited||0)+(s.transient_provider_failure||0)+(s.dependency_blocked||0);return `<div class="pool-row"><h3>${esc(row.job_id)} <span class="tag">${esc(row.status)}</span></h3><div class="muted">${esc((row.tasks||[]).join(', ')||'task unknown')} · ${esc((row.destinations||[]).join(', ')||'destination unknown')}</div><div class="route-state"><strong>${esc(s.completed||0)}/${esc(s.total||0)} completed</strong> · queued ${esc(s.queued||0)} · running ${esc(s.running||0)} · failed ${esc(s.failed||0)}<br><span class="${issue?'warn':''}">429 ${esc(s.rate_limited||0)} · transient/5xx ${esc(s.transient_provider_failure||0)} · dependency blocked ${esc(s.dependency_blocked||0)}</span><br><span class="revision">${esc(row.updated_at||'')}</span></div></div>`}).join('')||'<div class="empty">没有持久化在线模型批次</div>'}
async function refreshBatches(){try{batchState=await request('/api/model-batches');renderBatches();message(`已刷新 ${batchState.count||0} 个批次；Provider 限流由 LiteLLM 管理`,true,"batchStatus")}catch(error){message(error.message,false,"batchStatus")}}
const BLOCKER_LABELS={implement_runtime_adapter:"\u5f85\u5b9e\u73b0\u56fa\u5b9a\u534f\u8bae\u9002\u914d",create_profile:"\u672a\u5efa\u7acb profile",select_model:"\u672a\u9009\u62e9\u6a21\u578b",add_credential:"\u672a\u4fdd\u5b58\u51ed\u636e",configure_route:"\u672a\u914d\u7f6e\u8def\u7531",approve_destination:"\u76ee\u7684\u5730\u672a\u8fdb\u5165 allowlist",create_consent:"\u5f85\u751f\u6210 consent"};
function renderFreeOnboarding(){
 const host=$("freeOnboarding");const rows=state.free_screening_onboarding?.entries||[];
 host.innerHTML=rows.map(row=>{
  const blockers=(row.blockers||[]).map(x=>BLOCKER_LABELS[x]||x).join(" / ");
  const tags=(row.capabilities||[]).map(x=>'<span class="tag">'+esc(x)+'</span>').join("")+'<span class="tag">'+esc(row.runtime_integration)+'</span>';
  const links='<div class="onboarding-links"><a href="'+esc(row.account_url)+'" target="_blank" rel="noopener noreferrer">Account</a><a href="'+esc(row.credential_url)+'" target="_blank" rel="noopener noreferrer">Credential</a><a href="'+esc(row.documentation_url)+'" target="_blank" rel="noopener noreferrer">Docs</a></div>';
  const models=(row.profile_templates||[]).map(template=>{const suggested=(template.recommended_tasks||[]).join(", ");return '<div class="onboarding-model"><strong>'+esc(template.name)+'</strong><br><code>'+esc(template.model)+'</code><br>'+esc((template.capabilities||[]).join(" + "))+' / '+esc(template.protocol||"")+(suggested?'<br>Suggested tasks: <code>'+esc(suggested)+'</code>':"")+(template.note?'<br><span class="warn">'+esc(template.note)+'</span>':"")+'</div>'}).join("");
  const contract=row.prefill_contract||{};const provenance=contract.contract_id?'<div class="muted">字段契约：<code>'+esc(contract.contract_id)+'</code><br>官方核对：'+esc(contract.last_verified_at||'unknown')+' · SHA '+esc(String(contract.contract_sha256||'').slice(0,12))+'…</div>':'';
  const installed=(row.installed_profile_ids||[]).length;const expected=(row.expected_profile_ids||[]).length;
  const actionLabel=row.profile_saved?(row.credential_configured?"更新 API Key":"填写 API Key"):"填写 API Key 并安装预设";
  const action=row.key_once_ready?'<div class="onboarding-key"><input type="password" autocomplete="new-password" data-bundle-key placeholder="只在本机加密保存；提交后立即清空"><button type="button" class="primary" data-install-bundle="'+esc(row.id)+'">'+actionLabel+'</button></div>':'<button type="button" disabled>需先在供应商控制台选定精确模型</button>';
  const probe=row.key_once_ready&&row.profile_saved&&row.credential_configured?'<div class="onboarding-key"><button type="button" data-probe-bundle="'+esc(row.id)+'">只读检查 Key 与模型目录</button><span class="muted" data-probe-result></span></div>':'';
  return '<article class="onboarding-row"><h3>'+esc(row.label)+' <span class="tag">'+esc(row.priority)+'</span></h3><div>'+tags+'</div><div class="readiness"><strong>'+esc(row.status)+'</strong> / profiles '+installed+'/'+expected+'<br><span class="warn">'+esc(blockers)+'</span></div>'+provenance+models+'<div class="muted">'+esc(row.free_tier_note)+'</div><div class="muted">'+esc(row.data_boundary)+'</div>'+links+action+probe+'</article>';
 }).join("")||'<div class="empty">No onboarding metadata</div>';
 host.querySelectorAll("[data-install-bundle]").forEach(node=>node.addEventListener("click",()=>installOnboarding(node.dataset.installBundle,node)));
 host.querySelectorAll("[data-probe-bundle]").forEach(node=>node.addEventListener("click",()=>probeOnboarding(node.dataset.probeBundle,node)));
}
async function installOnboarding(id,button){
 const card=button.closest(".onboarding-row");const keyNode=card?.querySelector("[data-bundle-key]");const apiKey=String(keyNode?.value||"").trim();
 if(!apiKey){message("请先填写该供应商的 API Key",false);keyNode?.focus();return}
 button.disabled=true;
 try{
  const data=await request("/api/onboarding/"+encodeURIComponent(id),{method:"PUT",body:JSON.stringify({api_key:apiKey})});
  if(keyNode)keyNode.value="";state=data.settings;render();
  const count=data.settings.last_onboarding_install?.profile_count||0;
   message("已加密保存 API Key，并安装 "+count+" 个精确模型预设；未联网、未改路由、未授权外发");
 }catch(error){if(keyNode)keyNode.value="";message(error.message,false);button.disabled=false}
}
async function probeOnboarding(id,button){
 const card=button.closest(".onboarding-row");const resultNode=card?.querySelector("[data-probe-result]");button.disabled=true;if(resultNode)resultNode.textContent="正在读取供应商模型目录…";
 try{
  const data=await request("/api/onboarding-catalog/"+encodeURIComponent(id),{method:"POST",body:JSON.stringify({execute:true})});
  const probe=data.probe||{};const visible=(probe.catalog_entries||[]).filter(row=>row.visible).length;const total=(probe.catalog_entries||[]).length;
  if(resultNode)resultNode.textContent=probe.status+" · 可见 "+visible+"/"+total+" · catalog "+(probe.catalog_count??"?");
  message("模型目录检查完成；1 次元数据调用，0 次推理，0 个文件读取或上传。",Boolean(probe.ok));
 }catch(error){if(resultNode)resultNode.textContent=error.message;message(error.message,false)}
 finally{button.disabled=false}
}

const renderBase=render;render=function(){renderBase();renderFreeOnboarding()};
function renderScreeningLab(){const lab=state.model_screening_lab||{};const scenarioNode=$("screeningScenario");const selected=scenarioNode.value;const scenarios=lab.scenarios||[];scenarioNode.innerHTML=scenarios.map(row=>'<option value="'+esc(row.id)+'">'+esc(row.label)+' / '+esc(row.status)+'</option>').join("");if(scenarios.some(row=>row.id===selected))scenarioNode.value=selected;const criteria=lab.criteria||{};$("screeningCriteria").innerHTML=Object.entries(criteria).map(entry=>{const key=entry[0];const row=entry[1]||{};return '<span class="tag">'+esc(key)+' '+esc(row.weight)+'%'+(row.requires_real_result?' / real evidence':' / policy review')+'</span>'}).join("")+'<span class="tag">total '+esc(lab.criteria_weight_total||0)+'%</span>';const candidates=lab.candidates||[];$("screeningCandidates").innerHTML=candidates.map(row=>'<article class="onboarding-row"><h3>'+esc(row.label)+' <span class="tag">'+esc(row.priority)+'</span></h3><div>'+(row.capabilities||[]).map(capability=>'<span class="tag">'+esc(capability)+'</span>').join("")+'<span class="tag">'+esc(row.runtime_integration)+'</span></div><div class="readiness"><strong>'+esc(row.readiness_status)+'</strong><br>profiles '+esc(row.configured_profile_count||0)+' / consent '+esc(row.consent_status||"not_checked")+'</div><div class="muted">Offline contract simulation only; real quality ranking requires reviewed results.</div></article>').join("")||'<div class="empty">No screening candidates</div>'}
const renderWithOnboarding=render;render=function(){renderWithOnboarding();renderScreeningLab()};
function routePayload(){const route_pools=(state.route_pools||[]).map(pool=>{const node=document.querySelector(`[data-pool="${pool.id}"] .pool-deployments`);return {...pool,deployments:String(node?.value||'').split(',').map(x=>x.trim()).filter(Boolean)}});const route_bindings={};document.querySelectorAll('.route-row').forEach(node=>{const local=node.querySelector('.route-local').value;const remote=node.querySelector('.route-remote').value;if(local||remote)route_bindings[node.dataset.task]={default_location:node.querySelector('.route-default').value,local_pool_id:local,remote_pool_id:remote}});return {route_pools,route_bindings}}
async function load(){try{state=await request('/api/settings');render();if(activeId&&state.profiles.some(x=>x.id===activeId))edit(activeId);await refreshBatches();message("配置、路由与批次状态已刷新")}catch(error){message(error.message,false)}}
$("profileForm").addEventListener("submit",async event=>{event.preventDefault();try{const data=await request('/api/profile',{method:'PUT',body:JSON.stringify(payload())});state=data.settings;activeId=payload().profile.id||state.profiles.find(x=>x.name===$("name").value)?.id||"";render();if(activeId)edit(activeId);message("已安全保存到本机；尚未授权外发")}catch(error){message(error.message,false)}});
$("validateBtn").addEventListener("click",async()=>{try{await request('/api/validate',{method:'POST',body:JSON.stringify(payload())});message("结构与 URL 策略通过；未联网、未保存、未授权")}catch(error){message(error.message,false)}});
$("deleteBtn").addEventListener("click",async()=>{if(!activeId||!confirm("删除此 profile 及其本地凭据？"))return;try{const data=await request('/api/profile/'+encodeURIComponent(activeId),{method:'DELETE',body:JSON.stringify({})});state=data.settings;reset();message("profile 与本地凭据已删除")}catch(error){message(error.message,false)}});
$("saveRoutes").addEventListener("click",async()=>{try{const data=await request('/api/routes',{method:'PUT',body:JSON.stringify(routePayload())});state=data.settings;render();message("路由已保存；revision 已按内容重新计算，远程 consent 需重新匹配",true,"routeStatus")}catch(error){message(error.message,false,"routeStatus")}});
$("newBtn").addEventListener("click",reset);$("refreshBtn").addEventListener("click",load);$("refreshBatches").addEventListener("click",refreshBatches);$("provider").addEventListener("change",()=>{$("baseUrl").value="";$("model").value="";applyPreset()});$("toggleKey").addEventListener("click",()=>{$("apiKey").type=$("apiKey").type==="password"?"text":"password"});
$("runScreening").addEventListener("click",async()=>{try{message("Running loopback-only contract simulation...",true,"screeningStatus");const data=await request("/api/screening/simulate",{method:"POST",body:JSON.stringify({task:$("screeningTask").value,scenario:$("screeningScenario").value})});$("screeningResult").textContent=JSON.stringify(data.simulation,null,2);message("Simulation complete; no provider request, artifact read, payload, or credential access occurred.",true,"screeningStatus")}catch(error){message(error.message,false,"screeningStatus")}});
initOptions();reset();load();
</script>
</body></html>'''
    return (
        template.replace("__CSRF_TOKEN__", token_json)
        .replace("__PROVIDERS__", providers_json)
        .replace("__TASKS__", tasks_json)
    )

def _is_loopback_host(value: str) -> bool:
    host = str(value or "").strip().lower()
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _require_loopback_host(value: str) -> None:
    if not _is_loopback_host(value):
        raise ValueError("model API settings UI must bind to a loopback host")


def _service_defaults() -> tuple[str, int]:
    configured = load_model_api_settings_ui_config()
    return str(configured["host"]), int(configured["port"])


def main(argv: list[str] | None = None) -> None:
    default_host, default_port = _service_defaults()
    parser = argparse.ArgumentParser(description="Run the loopback-only VKP model API settings UI")
    parser.add_argument("--host", default=os.environ.get("VKP_MODEL_API_SETTINGS_HOST", default_host))
    parser.add_argument("--port", type=int, default=int(os.environ.get("VKP_MODEL_API_SETTINGS_PORT", default_port)))
    parser.add_argument("--settings-path", default=os.environ.get("VKP_MODEL_API_SETTINGS_PATH", ""))
    parser.add_argument("--secrets-path", default=os.environ.get("VKP_MODEL_API_SECRETS_PATH", ""))
    args = parser.parse_args(argv)
    server = build_server(
        host=args.host,
        port=args.port,
        settings_path=args.settings_path or None,
        secrets_path=args.secrets_path or None,
    )
    address, port = server.server_address[:2]
    print(f"VKP model API settings UI: http://{address}:{port}/", flush=True)
    print("Loopback only. API keys are stored with Windows DPAPI.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

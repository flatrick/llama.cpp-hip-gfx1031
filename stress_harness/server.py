from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Mapping

from .config import StressConfig
from .models import RequestMetrics, RuntimeInfo
from .runtime import ContainerLogReader, ContainerRuntimeInspector


class RequestWatchdog:
    def __init__(
        self,
        client: "LlamaServerClient",
        config: StressConfig,
        runtime_inspector: ContainerRuntimeInspector,
        runtime_info: RuntimeInfo,
        log_reader: ContainerLogReader | None,
    ) -> None:
        self._client = client
        self._config = config
        self._runtime_inspector = runtime_inspector
        self._runtime_info = runtime_info
        self._log_reader = log_reader

    def wait(self, done: threading.Event, started_at: float) -> None:
        last_signal = started_at
        last_slots = self._client.slots_snapshot()
        last_log_seq = self._log_reader.line_count() if self._log_reader else 0

        while not done.wait(self._config.watchdog_poll_s):
            now = time.monotonic()
            if now - started_at > self._config.request_timeout:
                raise TimeoutError(
                    f"request exceeded hard timeout of {self._config.request_timeout}s"
                )

            running = self._runtime_inspector.container_running(self._runtime_info)
            if running is False:
                raise RuntimeError("container exited while request was in flight")

            signal_seen = False
            current_slots = self._client.slots_snapshot()
            if current_slots is not None:
                if current_slots != last_slots:
                    signal_seen = True
                elif any(item[2] for item in current_slots):
                    signal_seen = True
                last_slots = current_slots

            if self._log_reader:
                current_log_seq = self._log_reader.line_count()
                if current_log_seq != last_log_seq:
                    signal_seen = True
                    last_log_seq = current_log_seq

            if signal_seen:
                last_signal = now
                continue

            if self._config.stall_timeout > 0 and now - last_signal > self._config.stall_timeout:
                raise TimeoutError(
                    f"no server progress signal for {self._config.stall_timeout}s "
                    f"(watchdog poll {self._config.watchdog_poll_s:g}s)"
                )


class LlamaServerClient:
    def __init__(self, config: StressConfig) -> None:
        self._config = config
        self._slots_url = self._join_api("/slots")
        self._props_url = self._join_api("/props")
        self._health_url = self._join_api("/health")

    def _join_api(self, path: str) -> str:
        base = self._config.api_url.split("/v1/", 1)[0]
        return f"{base}{path}"

    def _fetch_json(self, url: str, timeout: int = 3):
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.loads(response.read())

    @staticmethod
    def _as_mapping(value: Any) -> dict[str, Any]:
        if isinstance(value, Mapping):
            return dict(value)
        return {}

    def server_healthy(self) -> bool:
        try:
            urllib.request.urlopen(self._health_url, timeout=5)
            return True
        except Exception:
            return False

    def server_ctx_size(self) -> int:
        try:
            slots = self._fetch_json(self._slots_url, timeout=3)
            if slots and isinstance(slots, list) and "n_ctx" in slots[0]:
                return int(slots[0]["n_ctx"])
        except Exception:
            pass

        try:
            data = self._fetch_json(self._props_url, timeout=3)
            data_map = self._as_mapping(data)
            settings = self._as_mapping(data_map.get("default_generation_settings", {}))
            if "n_ctx" in settings:
                return int(settings["n_ctx"])
            if "n_ctx" in data_map:
                return int(data_map["n_ctx"])
        except Exception:
            pass

        return self._config.ctx_size_fallback

    def slots_snapshot(self) -> tuple[tuple[object, ...], ...] | None:
        try:
            slots = self._fetch_json(self._slots_url, timeout=3)
        except Exception:
            return None
        if not isinstance(slots, list):
            return None

        snapshot: list[tuple[object, ...]] = []
        for slot in slots:
            slot_map = self._as_mapping(slot)
            if not slot_map:
                continue
            next_token = self._as_mapping(slot_map.get("next_token", {}))
            snapshot.append((
                slot_map.get("id"),
                slot_map.get("id_task"),
                bool(slot_map.get("is_processing")),
                next_token.get("n_decoded", 0),
            ))
        return tuple(snapshot)

    def send_request(
        self,
        prompt: str,
        runtime_inspector: ContainerRuntimeInspector,
        runtime_info: RuntimeInfo,
        log_reader: ContainerLogReader | None,
        system: str | None = None,
    ) -> RequestMetrics:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = json.dumps({
            "model": "local",
            "messages": messages,
            "max_tokens": self._config.max_tokens,
            "temperature": 0,
        }).encode()

        request = urllib.request.Request(
            self._config.api_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        started_at = time.monotonic()
        result: dict[str, object] = {}
        done = threading.Event()

        def _run_request() -> None:
            try:
                with urllib.request.urlopen(request, timeout=self._config.request_timeout) as response:
                    result["body"] = json.loads(response.read())
            except Exception as error:
                result["error"] = error
            finally:
                result["elapsed_s"] = time.monotonic() - started_at
                done.set()

        threading.Thread(target=_run_request, daemon=True).start()
        watchdog = RequestWatchdog(
            client=self,
            config=self._config,
            runtime_inspector=runtime_inspector,
            runtime_info=runtime_info,
            log_reader=log_reader,
        )
        watchdog.wait(done, started_at)

        if "error" in result:
            raise result["error"]  # type: ignore[misc]
        body = result.get("body")
        if body is None:
            raise RuntimeError("request completed without a JSON body")
        elapsed_s = float(result.get("elapsed_s", time.monotonic() - started_at))
        return RequestMetrics.from_response(body, elapsed_s)

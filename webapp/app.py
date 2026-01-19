from __future__ import annotations

import os
import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

APP_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = APP_ROOT / "webapp_config.yml"
STATIC_DIR = APP_ROOT / "static"
INDEX_PATH = STATIC_DIR / "index.html"

app = FastAPI(title="ControlNet Trainer")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class RunRequest(BaseModel):
    config_path: Optional[str] = None


@dataclass
class RunState:
    run_id: str
    model_id: str
    status: str
    start_time: str
    end_time: Optional[str]
    return_code: Optional[int]
    log_path: str
    config_path: str
    command: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "model_id": self.model_id,
            "status": self.status,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "return_code": self.return_code,
            "log_path": self.log_path,
            "config_path": self.config_path,
            "command": self.command,
        }


run_lock = threading.Lock()
runs: Dict[str, RunState] = {}
active_run_id: Optional[str] = None
last_run_id: Optional[str] = None
active_process: Optional[subprocess.Popen] = None


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    if not INDEX_PATH.exists():
        raise HTTPException(status_code=500, detail="UI assets missing.")
    return HTMLResponse(INDEX_PATH.read_text(encoding="utf-8"))


@app.get("/api/models")
def list_models() -> JSONResponse:
    config = _load_webapp_config()
    models = _get_models(config)
    payload = []
    for model in models.values():
        payload.append(
            {
                "id": model["id"],
                "name": model.get("name", model["id"]),
                "description": model.get("description", ""),
                "default_config": model.get("default_config", ""),
            }
        )
    return JSONResponse({"models": payload})


@app.get("/api/status")
def status() -> JSONResponse:
    with run_lock:
        run = _get_last_run_locked()
        if run is None:
            return JSONResponse({"status": "idle", "run": None})
        state = "running" if active_run_id == run.run_id else run.status
        return JSONResponse({"status": state, "run": run.to_dict()})


@app.post("/api/run/{model_id}")
def run_model(model_id: str, request: RunRequest) -> JSONResponse:
    config = _load_webapp_config()
    model = _get_models(config).get(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="Unknown model.")
    run_state = _launch_run(model, request.config_path, config)
    return JSONResponse({"status": "running", "run": run_state.to_dict()})


@app.get("/api/logs")
def logs(
    run_id: Optional[str] = Query(default=None),
    offset: int = Query(default=0, ge=0),
) -> JSONResponse:
    with run_lock:
        run = _get_run_locked(run_id)
    if run is None:
        return JSONResponse({"data": "", "offset": offset})
    log_path = Path(run.log_path)
    if not log_path.exists():
        return JSONResponse({"data": "", "offset": offset})
    data, new_offset = _read_log_chunk(log_path, offset)
    return JSONResponse({"data": data, "offset": new_offset})


def _load_webapp_config() -> Dict[str, Any]:
    if not CONFIG_PATH.exists():
        raise HTTPException(status_code=500, detail="Missing webapp_config.yml.")
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    return config


def _get_models(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    models = config.get("models", [])
    model_map: Dict[str, Dict[str, Any]] = {}
    for model in models:
        if "id" not in model:
            continue
        model_map[model["id"]] = model
    return model_map


def _resolve_repo_root(config: Dict[str, Any]) -> Path:
    repo_root = config.get("paths", {}).get("repo_root")
    if repo_root:
        return Path(repo_root)
    return APP_ROOT.parent


def _resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def _timestamp() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _launch_run(
    model: Dict[str, Any], config_override: Optional[str], config: Dict[str, Any]
) -> RunState:
    global active_run_id
    global last_run_id
    global active_process

    with run_lock:
        if active_process is not None and active_process.poll() is None:
            raise HTTPException(status_code=409, detail="A run is already in progress.")

        repo_root = _resolve_repo_root(config)
        runs_dir_value = config.get("paths", {}).get("runs_dir", "webapp_runs")
        runs_dir = _resolve_path(repo_root, runs_dir_value)
        runs_dir.mkdir(parents=True, exist_ok=True)

        default_config = model.get("default_config")
        if not default_config and not config_override:
            raise HTTPException(status_code=400, detail="No config file provided.")
        config_path_value = config_override or default_config
        if config_path_value is None:
            raise HTTPException(status_code=400, detail="No config file provided.")
        config_path = _resolve_path(repo_root, str(config_path_value))
        if not config_path.exists():
            raise HTTPException(status_code=400, detail="Config file not found.")

        script_value = model.get("script")
        if not script_value:
            raise HTTPException(status_code=400, detail="Model script missing.")
        script_path = _resolve_path(repo_root, script_value)
        if not script_path.exists():
            raise HTTPException(status_code=400, detail="Model script not found.")

        run_id = f"{model['id']}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        log_path = runs_dir / f"{run_id}.log"
        command = [
            sys.executable,
            str(script_path),
            "--config",
            str(config_path),
        ]
        for arg in model.get("args", []):
            command.append(str(arg))

        env = os.environ.copy()
        env["PYTHONPATH"] = str(repo_root)

        log_file = log_path.open("ab")
        process = subprocess.Popen(
            command,
            cwd=str(repo_root),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
        )
        log_file.close()

        run_state = RunState(
            run_id=run_id,
            model_id=model["id"],
            status="running",
            start_time=_timestamp(),
            end_time=None,
            return_code=None,
            log_path=str(log_path),
            config_path=str(config_path),
            command=command,
        )
        runs[run_id] = run_state
        active_run_id = run_id
        last_run_id = run_id
        active_process = process

    thread = threading.Thread(
        target=_watch_process, args=(run_id, process), daemon=True
    )
    thread.start()

    return run_state


def _watch_process(run_id: str, process: subprocess.Popen) -> None:
    global active_run_id
    global active_process

    return_code = process.wait()
    with run_lock:
        run_state = runs.get(run_id)
        if run_state is None:
            return
        run_state.return_code = return_code
        run_state.end_time = _timestamp()
        run_state.status = "finished" if return_code == 0 else "failed"
        if active_run_id == run_id:
            active_run_id = None
            active_process = None


def _get_run_locked(run_id: Optional[str]) -> Optional[RunState]:
    if run_id:
        return runs.get(run_id)
    return _get_last_run_locked()


def _get_last_run_locked() -> Optional[RunState]:
    if active_run_id and active_run_id in runs:
        return runs[active_run_id]
    if last_run_id and last_run_id in runs:
        return runs[last_run_id]
    return None


def _read_log_chunk(path: Path, offset: int, max_bytes: int = 65536) -> tuple[str, int]:
    with path.open("rb") as handle:
        handle.seek(offset)
        data = handle.read(max_bytes)
        new_offset = handle.tell()
    return data.decode("utf-8", errors="replace"), new_offset

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
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


class StopRequest(BaseModel):
    run_id: Optional[str] = None


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
        if active_run_id == run.run_id:
            state = "stopping" if run.status == "stopping" else "running"
        else:
            state = run.status
        return JSONResponse({"status": state, "run": run.to_dict()})


@app.post("/api/run/{model_id}")
def run_model(model_id: str, request: RunRequest) -> JSONResponse:
    config = _load_webapp_config()
    model = _get_models(config).get(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="Unknown model.")
    run_state = _launch_run(model, request.config_path, config)
    return JSONResponse({"status": "running", "run": run_state.to_dict()})


@app.get("/api/sample")
def sample(run_id: Optional[str] = Query(default=None)) -> JSONResponse:
    with run_lock:
        run = _get_run_locked(run_id)
    if run is None:
        return JSONResponse({"available": False})
    config = _load_webapp_config()
    sample_path = _get_latest_sample_path(run, config)
    if sample_path is None:
        return JSONResponse({"available": False})
    mtime = sample_path.stat().st_mtime
    updated_at = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
    image_url = f"/api/sample_image?run_id={run.run_id}&ts={int(mtime)}"
    return JSONResponse(
        {"available": True, "updated_at": updated_at, "image_url": image_url}
    )


@app.get("/api/sample_image")
def sample_image(run_id: Optional[str] = Query(default=None)) -> FileResponse:
    with run_lock:
        run = _get_run_locked(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found.")
    config = _load_webapp_config()
    sample_path = _get_latest_sample_path(run, config)
    if sample_path is None:
        raise HTTPException(status_code=404, detail="No samples available.")
    return FileResponse(sample_path, media_type="image/png")


@app.post("/api/stop")
def stop_run(request: StopRequest) -> JSONResponse:
    with run_lock:
        run = _get_run_locked(request.run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found.")
        if active_run_id != run.run_id or active_process is None:
            raise HTTPException(status_code=409, detail="No active run to stop.")
        if active_process.poll() is not None:
            raise HTTPException(status_code=409, detail="Run is already finished.")
        run.status = "stopping"
        process = active_process

    _request_stop(process)
    return JSONResponse({"status": "stopping", "run": run.to_dict()})


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
        env["CONTROLNET_RUN_ID"] = run_id

        log_file = log_path.open("ab")
        process = subprocess.Popen(
            command,
            cwd=str(repo_root),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
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
        if run_state.status == "stopping":
            run_state.status = "stopped"
        else:
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


def _get_latest_sample_path(
    run: RunState, config: Dict[str, Any]
) -> Optional[Path]:
    repo_root = _resolve_repo_root(config)
    config_path = Path(run.config_path)
    if not config_path.exists():
        return None
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            run_config = yaml.safe_load(handle) or {}
    except yaml.YAMLError:
        return None
    train_cfg = run_config.get("train_params", {})
    output_dir_value = train_cfg.get("output_dir", "outputs")
    output_dir = _resolve_path(repo_root, str(output_dir_value))
    if output_dir.name != "ddpm":
        output_dir = output_dir / "ddpm"
    samples_dir = output_dir / "ddpm_samples"
    if run.run_id:
        samples_dir = samples_dir / run.run_id
    if not samples_dir.exists():
        return None
    sample_files = sorted(
        samples_dir.glob("*.png"), key=lambda path: path.stat().st_mtime
    )
    if not sample_files:
        return None
    return sample_files[-1]


def _request_stop(process: subprocess.Popen, timeout: float = 5.0) -> None:
    try:
        if process.poll() is not None:
            return
        if process.pid and hasattr(os, "killpg"):
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
    except OSError:
        return

    def _ensure_killed() -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if process.poll() is not None:
                return
            time.sleep(0.2)
        try:
            if process.pid and hasattr(os, "killpg"):
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except OSError:
            return

    threading.Thread(target=_ensure_killed, daemon=True).start()

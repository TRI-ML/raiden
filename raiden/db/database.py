"""pysonDB-backed metadata store for Raiden.

Collections (one JSON file each under ~/.config/raiden/db/):
  - demonstrations.json
  - teachers.json
  - tasks.json
  - calibration_results.json
  - camera_configs.json
"""

import fcntl
import json
import shutil
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, Generator, List, Optional

from pysondb import getDb

_DB_DIR = Path.home() / ".config" / "raiden" / "db"
_COLLECTIONS = [
    "demonstrations",
    "teachers",
    "tasks",
    "calibration_results",
    "camera_configs",
]
_instance: Optional["RaidenDB"] = None

_EMPTY_DB = {"data": [], "nextId": 1}


def _repair_file(path: Path) -> bool:
    """Check *path* for valid JSON; if corrupt, back it up and reinitialize.

    Returns True if a repair was performed.
    """
    try:
        with open(path) as f:
            data = json.load(f)
        # Also verify it has the expected pysonDB shape
        if not isinstance(data, dict) or "data" not in data:
            raise ValueError("missing 'data' key")
        return False  # file is fine
    except (json.JSONDecodeError, ValueError, OSError):
        ts = int(time.time())
        backup = path.with_suffix(f".bak{ts}.json")
        try:
            shutil.copy2(path, backup)
            print(f"  [DB] Corrupt file backed up → {backup.name}")
        except OSError:
            pass
        with open(path, "w") as f:
            json.dump(_EMPTY_DB, f, indent=3)
        print(f"  [DB] Reinitialized {path.name} (data lost — see backup)")
        return True


class RaidenDB:
    def __init__(self, db_dir: Path = _DB_DIR):
        db_dir.mkdir(parents=True, exist_ok=True)
        self._db_dir = db_dir
        self._lock_path = str(db_dir / ".lock")
        # Ensure lock file exists
        open(self._lock_path, "a").close()
        # In-process mutex (fast path when single process)
        self._mutex = threading.Lock()

        # Repair any corrupt collection files before opening them
        for name in _COLLECTIONS:
            _repair_file(db_dir / f"{name}.json")

        self.demonstrations = getDb(str(db_dir / "demonstrations.json"))
        self.teachers = getDb(str(db_dir / "teachers.json"))
        self.tasks = getDb(str(db_dir / "tasks.json"))
        self.calibration_results = getDb(str(db_dir / "calibration_results.json"))
        self.camera_configs = getDb(str(db_dir / "camera_configs.json"))

    @contextmanager
    def _lock(self) -> Generator[None, None, None]:
        """Acquire both in-process mutex and cross-process file lock."""
        with self._mutex:
            with open(self._lock_path, "r+") as lf:
                fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                except json.JSONDecodeError:
                    # A collection file was corrupt *inside* a locked section —
                    # repair all files and reload collections, then re-raise so
                    # the caller can retry or return a safe default.
                    for name in _COLLECTIONS:
                        repaired = _repair_file(self._db_dir / f"{name}.json")
                        if repaired:
                            setattr(
                                self,
                                name,
                                getDb(str(self._db_dir / f"{name}.json")),
                            )
                    raise
                finally:
                    fcntl.flock(lf.fileno(), fcntl.LOCK_UN)

    def repair(self) -> None:
        """Check and repair all collection files (call after detected corruption)."""
        with self._lock():
            for name in _COLLECTIONS:
                if _repair_file(self._db_dir / f"{name}.json"):
                    setattr(self, name, getDb(str(self._db_dir / f"{name}.json")))

    # ── Teachers ──────────────────────────────────────────────────────────────

    def add_teacher(self, name: str) -> int:
        with self._lock():
            return self.teachers.add(
                {"name": name, "created_at": datetime.now().isoformat()}
            )

    def get_teachers(self) -> List[Dict]:
        with self._lock():
            return self.teachers.getAll()

    def get_teacher_by_name(self, name: str) -> Optional[Dict]:
        with self._lock():
            for t in self.teachers.getAll():
                if t["name"] == name:
                    return t
            return None

    def update_teacher(self, teacher_id: int, name: str) -> None:
        with self._lock():
            self.teachers.updateById(teacher_id, {"name": name})

    def delete_teacher(self, teacher_id: int) -> None:
        with self._lock():
            self.teachers.deleteById(teacher_id)

    # ── Tasks ─────────────────────────────────────────────────────────────────

    def add_task(self, name: str, instruction: str) -> int:
        with self._lock():
            return self.tasks.add(
                {
                    "name": name,
                    "instruction": instruction,
                    "created_at": datetime.now().isoformat(),
                }
            )

    def get_tasks(self) -> List[Dict]:
        with self._lock():
            return self.tasks.getAll()

    def get_task_by_name(self, name: str) -> Optional[Dict]:
        with self._lock():
            for t in self.tasks.getAll():
                if t["name"] == name:
                    return t
            return None

    def update_task(self, task_id: int, name: str, instruction: str) -> None:
        with self._lock():
            self.tasks.updateById(task_id, {"name": name, "instruction": instruction})

    def delete_task(self, task_id: int) -> None:
        with self._lock():
            self.tasks.deleteById(task_id)

    # ── CalibrationResults ────────────────────────────────────────────────────

    def add_calibration_result(self, data: Dict, output_file: str) -> int:
        with self._lock():
            return self.calibration_results.add(
                {
                    "output_file": output_file,
                    "data": data,
                    "created_at": datetime.now().isoformat(),
                }
            )

    def get_calibration_results(self) -> List[Dict]:
        with self._lock():
            return self.calibration_results.getAll()

    def get_latest_calibration_result(self) -> Optional[Dict]:
        with self._lock():
            results = self.calibration_results.getAll()
            return results[-1] if results else None

    # ── CameraConfigs ─────────────────────────────────────────────────────────

    def snapshot_camera_config(self, data: Dict, config_file: str) -> int:
        """Snapshot camera config into DB; reuse existing entry if content is unchanged."""
        data_str = json.dumps(data, sort_keys=True)
        with self._lock():
            for cfg in self.camera_configs.getAll():
                if json.dumps(cfg.get("data", {}), sort_keys=True) == data_str:
                    return cfg["id"]
            return self.camera_configs.add(
                {
                    "config_file": config_file,
                    "data": data,
                    "created_at": datetime.now().isoformat(),
                }
            )

    def get_camera_configs(self) -> List[Dict]:
        with self._lock():
            return self.camera_configs.getAll()

    # ── Demonstrations ────────────────────────────────────────────────────────

    def add_demonstration(
        self,
        teacher_id: int,
        task_id: int,
        raw_data_path: str,
        camera_config_id: int,
        calibration_result_id: Optional[int],
    ) -> int:
        with self._lock():
            return self.demonstrations.add(
                {
                    "status": "pending",
                    "converted": False,
                    "raw_data_path": raw_data_path,
                    "converted_data_path": None,
                    "s3_data_path": None,
                    "teacher_id": teacher_id,
                    "task_id": task_id,
                    "camera_config_id": camera_config_id,
                    "calibration_result_id": calibration_result_id,
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat(),
                }
            )

    def get_demonstrations(self) -> List[Dict]:
        with self._lock():
            return self.demonstrations.getAll()

    def get_demonstrations_by_teacher(self, teacher_id: int) -> List[Dict]:
        with self._lock():
            return [
                d
                for d in self.demonstrations.getAll()
                if d.get("teacher_id") == teacher_id
            ]

    def get_demonstrations_by_task(self, task_id: int) -> List[Dict]:
        with self._lock():
            return [
                d for d in self.demonstrations.getAll() if d.get("task_id") == task_id
            ]

    def get_demonstration_by_raw_path(self, raw_data_path: str) -> Optional[Dict]:
        with self._lock():
            for d in self.demonstrations.getAll():
                if d.get("raw_data_path") == raw_data_path:
                    return d
            return None

    def get_demonstration_by_id(self, demo_id: int) -> Optional[Dict]:
        with self._lock():
            try:
                return self.demonstrations.getById(demo_id)
            except Exception:
                return None

    def update_demonstration(self, demo_id: int, **kwargs) -> None:
        kwargs["updated_at"] = datetime.now().isoformat()
        with self._lock():
            self.demonstrations.updateById(demo_id, kwargs)

    def delete_demonstration(self, demo_id: int) -> None:
        with self._lock():
            self.demonstrations.deleteById(demo_id)


def get_db() -> RaidenDB:
    global _instance
    if _instance is None:
        _instance = RaidenDB()
    return _instance


def reset_db() -> None:
    """Force the singleton to be recreated on next get_db() call.

    Useful after a detected corruption so the repaired files are reloaded.
    """
    global _instance
    _instance = None

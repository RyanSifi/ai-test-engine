"""
Job store en mémoire pour le suivi des tâches asynchrones (génération de tests).
Pour un usage multi-worker, remplacer par Redis ou une DB.
"""
import threading
import time
import uuid
from typing import Dict

_jobs: Dict[str, Dict] = {}
_jobs_lock = threading.Lock()
_JOB_TTL_SEC = 7200  # 2 heures

# Longueur de l'identifiant de job, en caractères hexadécimaux. 12 caractères
# = 48 bits : assez pour qu'une collision soit inatteignable en pratique, assez
# court pour rester lisible dans une URL de polling et dans les logs.
_JOB_ID_LENGTH = 12


def _new_job(**meta) -> str:
    """Crée un job, retourne son ID (12 hex chars). Purge les jobs expirés."""
    job_id = uuid.uuid4().hex[:_JOB_ID_LENGTH]
    now = time.time()
    with _jobs_lock:
        stale = [k for k, v in _jobs.items() if now - v.get("ts", 0) > _JOB_TTL_SEC]
        for k in stale:
            del _jobs[k]
        _jobs[job_id] = {"status": "pending", "ts": now, "result": None, "error": None, **meta}
    return job_id


def _update_job(job_id: str, **kwargs) -> None:
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update(kwargs)


def _get_job(job_id: str) -> Dict:
    with _jobs_lock:
        return dict(_jobs.get(job_id, {}))

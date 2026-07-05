from celery import Celery
import numpy as np
from systemictau import systemic_tau
import os

redis_url = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "systemictau_worker",
    broker=redis_url,
    backend=redis_url
)

@celery_app.task(name="compute_tau_heavy")
def compute_tau_heavy_task(data: list[list[float]], window_size: int, engine: str = "numba"):
    X = np.array(data)
    res = systemic_tau(X, window_size=window_size, engine=engine)
    # Returning global taus as list for JSON serialization
    return {"taus_global": res.taus_global.tolist()}

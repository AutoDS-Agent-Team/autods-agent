from celery import Celery
from app.core.config import get_settings

settings = get_settings()
celery_app = Celery("autods", broker=settings.celery_broker_url, backend=settings.celery_result_backend)
celery_app.conf.update(task_always_eager=settings.celery_task_always_eager, task_store_eager_result=True, task_track_started=True)
celery_app.autodiscover_tasks(["app.tasks"])

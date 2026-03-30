"""
ai_jobs/__init__.py — Job registry.

To add a new job type:
  1. Create ai_jobs/my_new_job.py with a class inheriting BaseJob.
  2. Import it here and add to JOB_REGISTRY.
"""

from ai_jobs.base_job import BaseJob
from ai_jobs.glossary_builder import GlossaryBuilderJob
from ai_jobs.st_job import SentenceTranslatorJob

JOB_REGISTRY: dict[str, type[BaseJob]] = {
    "glossary_builder": GlossaryBuilderJob,
    "translator":      SentenceTranslatorJob, 
    # "line_splitter":   LineSplitterJob,
}


def get_job_class(job_type: str) -> type[BaseJob]:
    cls = JOB_REGISTRY.get(job_type)
    if not cls:
        raise ValueError(f"Unknown job type: {job_type!r}. Registered: {list(JOB_REGISTRY)}")
    return cls
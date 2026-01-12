"""
Job queue manager for identity circuit factory.
Handles distributed processing of factory operations.
"""

import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from .database import CircuitDatabase, JobRecord

logger = logging.getLogger(__name__)


class JobType(Enum):
    """Types of jobs that can be processed."""

    SEED_GENERATION = "seed_generation"
    UNROLLING = "unrolling"
    POST_PROCESSING = "post_processing"
    DEBRIS_ANALYSIS = "debris_analysis"
    ML_FEATURE_EXTRACTION = "ml_feature_extraction"
    PARQUET_EXPORT = "parquet_export"


class JobStatus(Enum):
    """Job status values."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class JobProcessor:
    """Represents a job processor with its capabilities."""

    name: str
    job_types: List[JobType]
    max_concurrent_jobs: int = 1
    is_active: bool = True


class JobQueueManager:
    """Manages job queue for distributed factory processing."""

    def __init__(self, database: CircuitDatabase, max_workers: int = 4):
        self.database = database
        self.max_workers = max_workers
        self.processors: Dict[str, JobProcessor] = {}
        self.job_handlers: Dict[JobType, Callable] = {}
        self.running = False
        self.worker_threads: List[threading.Thread] = []
        self.stop_event = threading.Event()

        # Register default job handlers
        self._register_default_handlers()

    def _register_default_handlers(self):
        """Register default job handlers."""
        self.job_handlers[JobType.SEED_GENERATION] = self._handle_seed_generation
        self.job_handlers[JobType.UNROLLING] = self._handle_unrolling
        self.job_handlers[JobType.POST_PROCESSING] = self._handle_post_processing
        self.job_handlers[JobType.DEBRIS_ANALYSIS] = self._handle_debris_analysis
        self.job_handlers[JobType.ML_FEATURE_EXTRACTION] = (
            self._handle_ml_feature_extraction
        )
        self.job_handlers[JobType.PARQUET_EXPORT] = self._handle_parquet_export

    def register_processor(self, processor: JobProcessor):
        """Register a job processor."""
        self.processors[processor.name] = processor
        logger.info(f"Registered processor: {processor.name}")

    def register_job_handler(self, job_type: JobType, handler: Callable):
        """Register a custom job handler."""
        self.job_handlers[job_type] = handler
        logger.info(f"Registered handler for job type: {job_type.value}")

    def create_job(
        self, job_type: JobType, parameters: Dict[str, Any], priority: int = 0
    ) -> int:
        """Create a new job in the queue."""
        job = JobRecord(
            id=None,
            job_type=job_type.value,
            status=JobStatus.PENDING.value,
            priority=priority,
            parameters=parameters,
        )

        job_id = self.database.create_job(job)
        logger.info(f"Created job {job_id} of type {job_type.value}")
        return job_id

    def start_workers(self):
        """Start worker threads to process jobs."""
        if self.running:
            logger.warning("Job queue manager is already running")
            return

        self.running = True
        self.stop_event.clear()

        for i in range(self.max_workers):
            worker = threading.Thread(
                target=self._worker_loop, name=f"JobWorker-{i}", daemon=True
            )
            worker.start()
            self.worker_threads.append(worker)

        logger.info(f"Started {self.max_workers} job workers")

    def stop_workers(self):
        """Stop worker threads."""
        if not self.running:
            return

        self.running = False
        self.stop_event.set()

        for worker in self.worker_threads:
            worker.join(timeout=5.0)

        self.worker_threads.clear()
        logger.info("Stopped all job workers")

    def _worker_loop(self):
        """Main worker loop for processing jobs."""
        while self.running and not self.stop_event.is_set():
            try:
                # Get next pending job
                pending_jobs = self.database.get_pending_jobs(limit=1)

                if not pending_jobs:
                    time.sleep(1.0)  # No jobs, wait a bit
                    continue

                job = pending_jobs[0]

                # Check if we have a handler for this job type
                job_type = JobType(job.job_type)
                if job_type not in self.job_handlers:
                    logger.error(
                        f"No handler registered for job type: {job_type.value}"
                    )
                    self.database.update_job_status(
                        job.id,
                        JobStatus.FAILED.value,
                        error_message=f"No handler for job type: {job_type.value}",
                    )
                    continue

                # Process the job
                self._process_job(job, job_type)

            except Exception as e:
                logger.error(f"Error in worker loop: {e}")
                time.sleep(1.0)

    def _process_job(self, job: JobRecord, job_type: JobType):
        """Process a single job."""
        logger.info(f"Processing job {job.id} of type {job_type.value}")

        # Mark job as running
        self.database.update_job_status(job.id, JobStatus.RUNNING.value)

        try:
            # Get the handler for this job type
            handler = self.job_handlers[job_type]

            # Execute the handler
            result = handler(job.parameters)

            # Mark job as completed
            self.database.update_job_status(
                job.id, JobStatus.COMPLETED.value, result=result
            )

            logger.info(f"Completed job {job.id}")

        except Exception as e:
            logger.error(f"Failed to process job {job.id}: {e}")
            self.database.update_job_status(
                job.id, JobStatus.FAILED.value, error_message=str(e)
            )

    # Default job handlers
    def _handle_seed_generation(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle seed generation jobs."""
        width = parameters.get("width")
        length = parameters.get("length")
        max_attempts = parameters.get("max_attempts", 100)

        # This would integrate with the SeedGenerator
        # For now, return a placeholder result
        return {
            "width": width,
            "length": length,
            "seeds_generated": 0,
            "status": "placeholder",
        }

    def _handle_unrolling(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle circuit unrolling jobs."""
        dim_group_id = parameters.get("dim_group_id")
        seed_circuit_id = parameters.get("seed_circuit_id")
        unroll_types = parameters.get(
            "unroll_types", ["swaps", "rotations", "permutations"]
        )

        # This would integrate with the CircuitUnroller
        return {
            "dim_group_id": dim_group_id,
            "seed_circuit_id": seed_circuit_id,
            "circuits_generated": 0,
            "unroll_types": unroll_types,
        }

    def _handle_post_processing(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle post-processing jobs."""
        dim_group_id = parameters.get("dim_group_id")
        representative_circuit_id = parameters.get("representative_circuit_id")

        # This would integrate with the PostProcessor
        return {
            "dim_group_id": dim_group_id,
            "representative_circuit_id": representative_circuit_id,
            "simplifications_found": 0,
        }

    def _handle_debris_analysis(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle debris cancellation analysis jobs."""
        dim_group_id = parameters.get("dim_group_id")
        circuit_id = parameters.get("circuit_id")
        max_debris_gates = parameters.get("max_debris_gates", 5)

        # This would integrate with the DebrisCancellationManager
        return {
            "dim_group_id": dim_group_id,
            "circuit_id": circuit_id,
            "improvement_found": False,
            "non_triviality_score": 0.0,
        }

    def _handle_ml_feature_extraction(
        self, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle ML feature extraction jobs."""
        circuit_id = parameters.get("circuit_id")
        dim_group_id = parameters.get("dim_group_id")

        # This would integrate with ML feature extraction
        return {
            "circuit_id": circuit_id,
            "dim_group_id": dim_group_id,
            "features_extracted": 0,
        }

    def _handle_parquet_export(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle Parquet export jobs."""
        dim_group_id = parameters.get("dim_group_id")
        output_path = parameters.get("output_path")

        # This would integrate with Parquet export functionality
        return {
            "dim_group_id": dim_group_id,
            "output_path": output_path,
            "circuits_exported": 0,
        }

    # Job management methods
    def get_job_status(self, job_id: int) -> Optional[Dict[str, Any]]:
        """Get the status of a specific job."""
        # This would require adding a method to get a specific job by ID
        # For now, return a placeholder
        return None

    def cancel_job(self, job_id: int) -> bool:
        """Cancel a pending job."""
        try:
            self.database.update_job_status(
                job_id, JobStatus.CANCELLED.value, error_message="Job cancelled by user"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to cancel job {job_id}: {e}")
            return False

    def get_queue_stats(self) -> Dict[str, Any]:
        """Get statistics about the job queue."""
        stats = self.database.get_database_stats()

        # Add queue-specific stats
        queue_stats = {
            "active_workers": len([w for w in self.worker_threads if w.is_alive()]),
            "registered_processors": len(self.processors),
            "registered_handlers": len(self.job_handlers),
            "queue_running": self.running,
        }

        return {**stats, **queue_stats}

    def clear_completed_jobs(self, older_than_days: int = 7) -> int:
        """Clear completed jobs older than specified days."""
        # This would require adding a method to the database
        # For now, return a placeholder
        return 0


class AsyncJobQueueManager:
    """Async version of job queue manager for use with FastAPI."""

    def __init__(self, database: CircuitDatabase, max_workers: int = 4):
        self.database = database
        self.max_workers = max_workers
        self.running = False
        self.worker_tasks: List[asyncio.Task] = []
        self.stop_event = asyncio.Event()

        # Register default job handlers
        self.job_handlers: Dict[JobType, Callable] = {}
        self._register_default_handlers()

    def register_job_handler(self, job_type: JobType, handler: Callable):
        """Register a custom async job handler."""
        self.job_handlers[job_type] = handler
        logger.info(f"Registered async handler for job type: {job_type.value}")

    def _register_default_handlers(self):
        """Register default async job handlers."""
        self.job_handlers[JobType.SEED_GENERATION] = self._handle_seed_generation_async
        self.job_handlers[JobType.UNROLLING] = self._handle_unrolling_async
        self.job_handlers[JobType.POST_PROCESSING] = self._handle_post_processing_async
        self.job_handlers[JobType.DEBRIS_ANALYSIS] = self._handle_debris_analysis_async
        self.job_handlers[JobType.ML_FEATURE_EXTRACTION] = (
            self._handle_ml_feature_extraction_async
        )
        self.job_handlers[JobType.PARQUET_EXPORT] = self._handle_parquet_export_async

    async def start_workers(self):
        """Start async worker tasks."""
        if self.running:
            logger.warning("Async job queue manager is already running")
            return

        self.running = True
        self.stop_event.clear()

        for i in range(self.max_workers):
            task = asyncio.create_task(
                self._worker_loop_async(), name=f"AsyncJobWorker-{i}"
            )
            self.worker_tasks.append(task)

        logger.info(f"Started {self.max_workers} async job workers")

    async def stop_workers(self):
        """Stop async worker tasks."""
        if not self.running:
            return

        self.running = False
        self.stop_event.set()

        # Wait for all tasks to complete
        if self.worker_tasks:
            await asyncio.gather(*self.worker_tasks, return_exceptions=True)

        self.worker_tasks.clear()
        logger.info("Stopped all async job workers")

    async def _worker_loop_async(self):
        """Async worker loop for processing jobs."""
        while self.running and not self.stop_event.is_set():
            try:
                # Get next pending job
                pending_jobs = self.database.get_pending_jobs(limit=1)

                if not pending_jobs:
                    await asyncio.sleep(1.0)  # No jobs, wait a bit
                    continue

                job = pending_jobs[0]

                # Check if we have a handler for this job type
                job_type = JobType(job.job_type)
                if job_type not in self.job_handlers:
                    logger.error(
                        f"No handler registered for job type: {job_type.value}"
                    )
                    self.database.update_job_status(
                        job.id,
                        JobStatus.FAILED.value,
                        error_message=f"No handler for job type: {job_type.value}",
                    )
                    continue

                # Process the job
                await self._process_job_async(job, job_type)

            except Exception as e:
                logger.error(f"Error in async worker loop: {e}")
                await asyncio.sleep(1.0)

    async def _process_job_async(self, job: JobRecord, job_type: JobType):
        """Process a single job asynchronously."""
        logger.info(f"Processing job {job.id} of type {job_type.value}")

        # Mark job as running
        self.database.update_job_status(job.id, JobStatus.RUNNING.value)

        try:
            # Get the handler for this job type
            handler = self.job_handlers[job_type]

            # Execute the async handler
            result = await handler(job.parameters)

            # Mark job as completed
            self.database.update_job_status(
                job.id, JobStatus.COMPLETED.value, result=result
            )

            logger.info(f"Completed job {job.id}")

        except Exception as e:
            logger.error(f"Failed to process job {job.id}: {e}")
            self.database.update_job_status(
                job.id, JobStatus.FAILED.value, error_message=str(e)
            )

    # Async job handlers
    async def _handle_seed_generation_async(
        self, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle seed generation jobs asynchronously."""
        # Simulate async work
        await asyncio.sleep(0.1)
        return self._handle_seed_generation(parameters)

    async def _handle_unrolling_async(
        self, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle circuit unrolling jobs asynchronously."""
        await asyncio.sleep(0.1)
        return self._handle_unrolling(parameters)

    async def _handle_post_processing_async(
        self, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle post-processing jobs asynchronously."""
        await asyncio.sleep(0.1)
        return self._handle_post_processing(parameters)

    async def _handle_debris_analysis_async(
        self, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle debris cancellation analysis jobs asynchronously."""
        await asyncio.sleep(0.1)
        return self._handle_debris_analysis(parameters)

    async def _handle_ml_feature_extraction_async(
        self, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle ML feature extraction jobs asynchronously."""
        await asyncio.sleep(0.1)
        return self._handle_ml_feature_extraction(parameters)

    async def _handle_parquet_export_async(
        self, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle Parquet export jobs asynchronously."""
        await asyncio.sleep(0.1)
        return self._handle_parquet_export(parameters)

    # Reuse sync handlers for now
    def _handle_seed_generation(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        return JobQueueManager._handle_seed_generation(self, parameters)

    def _handle_unrolling(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        return JobQueueManager._handle_unrolling(self, parameters)

    def _handle_post_processing(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        return JobQueueManager._handle_post_processing(self, parameters)

    def _handle_debris_analysis(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        return JobQueueManager._handle_debris_analysis(self, parameters)

    def _handle_ml_feature_extraction(
        self, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        return JobQueueManager._handle_ml_feature_extraction(self, parameters)

    def _handle_parquet_export(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        return JobQueueManager._handle_parquet_export(self, parameters)

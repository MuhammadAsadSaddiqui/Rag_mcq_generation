import asyncio
from temporalio.client import Client
from temporalio.common import RetryPolicy
from temporalio.worker import Worker
from temporalio import workflow
from datetime import timedelta


@workflow.defn
class MCQGenerationWorkflow:
    @workflow.run
    async def run(self, input_data):
        return await workflow.execute_activity(
            "generate_mcq_activity",
            input_data,
            start_to_close_timeout=timedelta(minutes=10),
            retry_policy=RetryPolicy(maximum_attempts=3),
        )



async def start_worker():
    client = await Client.connect("localhost:7233")

    from activity import generate_mcq_activity

    worker = Worker(
        client,
        task_queue="mcq-queue",
        activities=[generate_mcq_activity],
        workflows=[MCQGenerationWorkflow],
    )

    await worker.run()
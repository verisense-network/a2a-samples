"""Butler Agent Executor for A2A"""

import logging
from typing import List
from pydantic import BaseModel

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    InternalError,
    InvalidParamsError,
    Task,
    TaskState,
    UnsupportedOperationError,
    Artifact,
)
from a2a.utils import (
    new_agent_text_message,
    new_task,
)
from a2a.utils.errors import ServerError

from butler.butler_agent import ButlerAgent
from app.rich_logging_config import get_rich_logger, log_info, log_error, log_warning

# Get logger with rich formatting
logger = get_rich_logger(__name__)


class ButlerAgentExecutor(AgentExecutor):
    """Butler AgentExecutor that orchestrates multiple agents."""

    def __init__(self, rpc_url: str = "https://rpc.beta.verisense.network"):
        self.agent = ButlerAgent(rpc_url=rpc_url)

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        error = self._validate_request(context)
        if error:
            raise ServerError(error=InvalidParamsError())

        query = context.get_user_input()

        # Extract conversation history from message parts
        conversation_parts = []
        if context.message and hasattr(context.message, "parts"):
            conversation_parts = context.message.parts

        task = context.current_task
        if not task:
            task = new_task(context.message)  # type: ignore
            await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue, task.id, task.contextId)

        try:
            async for item in self.agent.stream(
                query, task.contextId, conversation_parts
            ):
                is_task_complete = item["is_task_complete"]
                require_user_input = item["require_user_input"]
                artifact = item.get("artifact", None)

                logger.info(f"Butler executor: {item}")

                if not is_task_complete and not require_user_input:
                    if artifact:
                        # Type assertion to ensure artifact is an Artifact object
                        assert isinstance(
                            artifact, Artifact
                        ), f"Expected Artifact object, got {type(artifact)}"
                        await updater.add_artifact(
                            parts=artifact.parts,
                            artifact_id=artifact.artifactId,
                            name=artifact.name,
                            metadata=artifact.metadata,
                        )
                    else:
                        await updater.update_status(
                            TaskState.working,
                            new_agent_text_message(
                                item["content"],
                                task.contextId,
                                task.id,
                            ),
                        )
                elif require_user_input:
                    await updater.update_status(
                        TaskState.input_required,
                        new_agent_text_message(
                            item["content"],
                            task.contextId,
                            task.id,
                        ),
                        final=True,
                    )
                    break
                else:
                    await updater.update_status(
                        TaskState.completed,
                        new_agent_text_message(
                            item["content"],
                            task.contextId,
                            task.id,
                        ),
                    )
                    await updater.complete()
                    break

        except Exception as e:
            logger.error(f"An error occurred while executing butler: {e}")
            raise ServerError(error=InternalError()) from e

    def _validate_request(self, context: RequestContext) -> bool:
        return False

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise ServerError(error=UnsupportedOperationError())

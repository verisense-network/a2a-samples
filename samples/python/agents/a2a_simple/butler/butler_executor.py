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
        log_info(logger, "Butler executor started - received request from frontend")
        
        error = self._validate_request(context)
        if error:
            raise ServerError(error=InvalidParamsError())

        query = context.get_user_input()
        log_info(logger, f"Butler received query: {query}")

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
            log_info(logger, "Starting butler agent streaming")
            item_count = 0
            async for item in self.agent.stream(
                query, task.contextId, conversation_parts
            ):
                item_count += 1
                log_info(logger, f"Received stream item #{item_count}: is_complete={item.get('is_task_complete')}, requires_input={item.get('require_user_input')}, has_artifact={bool(item.get('artifact'))}")
                
                is_task_complete = item["is_task_complete"]
                require_user_input = item["require_user_input"]
                artifact = item.get("artifact", None)

                if not is_task_complete and not require_user_input:
                    if artifact:
                        # Type assertion to ensure artifact is an Artifact object
                        assert isinstance(
                            artifact, Artifact
                        ), f"Expected Artifact object, got {type(artifact)}"
                        log_info(logger, f"Butler sending artifact to frontend: {artifact.name} (id: {artifact.artifactId})")
                        try:
                            await updater.add_artifact(
                                parts=artifact.parts,
                                artifact_id=artifact.artifactId,
                                name=artifact.name,
                                metadata=artifact.metadata,
                            )
                            log_info(logger, f"Successfully sent artifact {artifact.name} to frontend")
                        except Exception as e:
                            log_error(logger, f"Failed to send artifact to frontend: {e}")
                            raise
                    else:
                        content = item["content"]
                        log_info(logger, f"Butler sending message to frontend (working): {content[:200]}..." if len(content) > 200 else f"Butler sending message to frontend (working): {content}")
                        try:
                            await updater.update_status(
                                TaskState.working,
                                new_agent_text_message(
                                    content,
                                    task.contextId,
                                    task.id,
                                ),
                            )
                            log_info(logger, "Successfully sent working message to frontend")
                        except Exception as e:
                            log_error(logger, f"Failed to send working message to frontend: {e}")
                            raise
                elif require_user_input:
                    content = item["content"]
                    log_info(logger, f"Butler sending input required message to frontend: {content[:200]}..." if len(content) > 200 else f"Butler sending input required message to frontend: {content}")
                    try:
                        await updater.update_status(
                            TaskState.input_required,
                            new_agent_text_message(
                                content,
                                task.contextId,
                                task.id,
                            ),
                            final=True,
                        )
                        log_info(logger, "Successfully sent input required message to frontend")
                    except Exception as e:
                        log_error(logger, f"Failed to send input required message to frontend: {e}")
                        raise
                    break
                else:
                    content = item["content"]
                    log_info(logger, f"Butler sending completion message to frontend: {content[:200]}..." if len(content) > 200 else f"Butler sending completion message to frontend: {content}")
                    try:
                        await updater.update_status(
                            TaskState.completed,
                            new_agent_text_message(
                                content,
                                task.contextId,
                                task.id,
                            ),
                        )
                        log_info(logger, "Successfully sent completion message to frontend")
                        await updater.complete()
                        log_info(logger, "Successfully marked task as complete")
                    except Exception as e:
                        log_error(logger, f"Failed to send completion message or mark task complete: {e}")
                        raise
                    break

        except Exception as e:
            logger.error(f"An error occurred while executing butler: {e}")
            raise ServerError(error=InternalError()) from e

    def _validate_request(self, context: RequestContext) -> bool:
        return False

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise ServerError(error=UnsupportedOperationError())

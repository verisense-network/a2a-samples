import logging

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    InternalError,
    InvalidParamsError,
    Part,
    Task,
    TaskState,
    TextPart,
    UnsupportedOperationError,
)
from a2a.utils import (
    new_agent_text_message,
    new_task,
)
from typing import List
from pydantic import BaseModel
from a2a.utils.errors import ServerError

from app.agent import PromptBasedAgent


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PromptAgentExecutor(AgentExecutor):
    """Prompt-based AgentExecutor."""

    def __init__(self, system_prompt: str = None, use_tools: List[BaseModel] = None):
        self.agent = PromptBasedAgent(system_prompt, use_tools)

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
                print("agent_executor.py:53", item)

                if not is_task_complete and not require_user_input:
                    await updater.update_status(
                        TaskState.working,
                        new_agent_text_message(
                            "⏲️ " + item["content"],
                            task.contextId,
                            task.id,
                        ),
                    )
                elif require_user_input:
                    await updater.update_status(
                        TaskState.input_required,
                        new_agent_text_message(
                            # question mark
                            "❓ " + item["content"] + "\\n",
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
                            "✅ " + item["content"] + "\n",
                            task.contextId,
                            task.id,
                        ),
                    )
                    await updater.complete()
                    break

        except Exception as e:
            logger.error(f"An error occurred while streaming the response: {e}")
            raise ServerError(error=InternalError()) from e

    def _validate_request(self, context: RequestContext) -> bool:
        return False

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise ServerError(error=UnsupportedOperationError())

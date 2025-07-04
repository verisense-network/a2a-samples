import asyncio
import base64
import os
import urllib

from uuid import uuid4

import httpx
from rich.logging import RichHandler
import logging

logging.basicConfig(
    level="DEBUG",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s - Line %(lineno)d",
    handlers=[RichHandler()],
)
logger = logging.getLogger(__name__)

from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    FilePart,
    FileWithBytes,
    GetTaskRequest,
    JSONRPCErrorResponse,
    Message,
    MessageSendConfiguration,
    MessageSendParams,
    Part,
    SendMessageRequest,
    SendStreamingMessageRequest,
    Task,
    TaskArtifactUpdateEvent,
    TaskQueryParams,
    TaskState,
    TaskStatusUpdateEvent,
    TextPart,
)

# from common.utils.push_notification_auth import PushNotificationReceiverAuth


async def call_agent(agent: str, contextId: str, prompt: str, timeout: int) -> Task:
    async with httpx.AsyncClient(timeout=timeout, headers={}) as httpx_client:
        card_resolver = A2ACardResolver(httpx_client, agent)
        card = await card_resolver.get_agent_card()
        print("======= Agent Card ========")
        print(card.model_dump_json(exclude_none=True, indent=2))
        client = A2AClient(httpx_client, agent_card=card)

        streaming = card.capabilities.streaming
        result = await completeTask(
            client,
            streaming,
            None,
            contextId,
            prompt,
        )
        return result


async def completeTask(
    client: A2AClient,
    streaming: bool,
    taskId: str | None,
    contextId: str,
    prompt: str,
) -> Task:
    message = Message(
        role="user",
        parts=[TextPart(text=prompt)],
        messageId=str(uuid4()),
        taskId=taskId,
        contextId=contextId,
    )

    payload = MessageSendParams(
        id=str(uuid4()),
        message=message,
        configuration=MessageSendConfiguration(
            acceptedOutputModes=["text"],
        ),
    )

    taskResult = None
    message = None
    if streaming:
        response_stream = client.send_message_streaming(
            SendStreamingMessageRequest(
                id=str(uuid4()),
                params=payload,
            )
        )
        async for result in response_stream:
            if isinstance(result.root, JSONRPCErrorResponse):
                logger.error(f"Error: {result.root.error}")
                raise Exception(result.root.error)
            event = result.root.result
            contextId = event.contextId
            if isinstance(event, Task):
                taskId = event.id
            elif isinstance(event, TaskStatusUpdateEvent) or isinstance(
                event, TaskArtifactUpdateEvent
            ):
                taskId = event.taskId
            elif isinstance(event, Message):
                message = event
            logger.info(f"stream event => {event.model_dump_json(exclude_none=True)}")
        # Upon completion of the stream. Retrieve the full task if one was made.
        if taskId:
            taskResult = await client.get_task(
                GetTaskRequest(
                    id=str(uuid4()),
                    params=TaskQueryParams(id=taskId),
                )
            )
            taskResult = taskResult.root.result
    else:
        try:
            # For non-streaming, assume the response is a task or message.
            event = await client.send_message(
                SendMessageRequest(
                    id=str(uuid4()),
                    params=payload,
                )
            )
            event = event.root.result
        except Exception as e:
            logger.error("Failed to complete the call", e)
        if not contextId:
            contextId = event.contextId
        if isinstance(event, Task):
            if not taskId:
                taskId = event.id
            taskResult = event
        elif isinstance(event, Message):
            message = event

    if message:
        logger.info(f"\n{message.model_dump_json(exclude_none=True)}")
        return contextId, taskId
    if taskResult:
        # Don't print the contents of a file.
        task_content = taskResult.model_dump_json(
            exclude={
                "history": {
                    "__all__": {
                        "parts": {
                            "__all__": {"file"},
                        },
                    },
                },
            },
            exclude_none=True,
            indent=2,
        )
        logger.info(f"\n{task_content}")
        ## if the result is that more input is required, loop again.
        state = TaskState(taskResult.status.state)
        if state.name == TaskState.input_required.name:
            raise Exception("Input required not implemented")
            return await completeTask(
                client,
                streaming,
                taskId,
                contextId,
                prompt,
            )
        ## task is complete
        return taskResult
    raise Exception("Task result is None")


async def main():
    try:
        task_result = await call_agent(
            # "http://34.57.6.105/p10004", "123", "get the price of btc price"
            # "http://34.57.6.105/p10002", "123", "what can you do?",
            "https://agent-coder-974618882715.europe-west1.run.app",
            "123",
            "write a code to calc fibonacci",
        )
    except Exception as e:
        logger.error(f"Error: {e}")
        return
    logger.info(f"Task ID: {task_result}")


if __name__ == "__main__":
    asyncio.run(main())

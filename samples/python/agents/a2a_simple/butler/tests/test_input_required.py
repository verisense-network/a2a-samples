#!/usr/bin/env python3
"""Test script for butler agent input_required functionality"""

import asyncio
import json
import logging
from uuid import uuid4
import httpx

from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    MessageSendParams,
    SendMessageRequest,
    SendStreamingMessageRequest,
    TaskStatusUpdateEvent,
    TaskState,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_butler_input_required():
    """Test butler agent with an agent that requires user input"""
    
    butler_url = "http://localhost:10000"  # Butler agent URL
    
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
        try:
            # Get butler agent card
            resolver = A2ACardResolver(client, butler_url)
            agent_card = await resolver.get_agent_card()
            
            a2a_client = A2AClient(client, agent_card=agent_card, url=butler_url)
            
            # Send initial message
            initial_message = "Please help me with a task that might require additional input"
            
            streaming_request = SendStreamingMessageRequest(
                id=str(uuid4()),
                params=MessageSendParams(
                    message={
                        "role": "user",
                        "parts": [{"kind": "text", "text": initial_message}],
                        "messageId": uuid4().hex,
                        "contextId": "test-context-123",
                    }
                ),
            )
            
            task_id = None
            context_id = "test-context-123"
            requires_input = False
            
            print(f"Sending initial message: {initial_message}")
            print("-" * 50)
            
            # Stream the response
            stream_response = a2a_client.send_message_streaming(streaming_request)
            async for chunk in stream_response:
                if hasattr(chunk, "root") and hasattr(chunk.root, "result"):
                    result = chunk.root.result
                    
                    if isinstance(result, TaskStatusUpdateEvent):
                        if hasattr(result, "taskId"):
                            task_id = result.taskId
                            
                        if hasattr(result, "status") and hasattr(result.status, "state"):
                            state = result.status.state
                            
                            if state == TaskState.input_required:
                                requires_input = True
                                print("\n🔔 Butler requires user input!")
                                
                                if hasattr(result.status, "message") and result.status.message:
                                    if hasattr(result.status.message, "parts"):
                                        for part in result.status.message.parts:
                                            if hasattr(part, "root") and hasattr(part.root, "text"):
                                                print(f"Message: {part.root.text}")
                                break
                            
                            elif state == TaskState.completed:
                                print("\n✅ Task completed!")
                                if hasattr(result.status, "message") and result.status.message:
                                    if hasattr(result.status.message, "parts"):
                                        for part in result.status.message.parts:
                                            if hasattr(part, "root") and hasattr(part.root, "text"):
                                                print(f"Result: {part.root.text}")
            
            # If input is required, send follow-up message
            if requires_input and task_id:
                print("\n" + "-" * 50)
                user_input = "Here is the additional information you requested"
                print(f"Sending user input: {user_input}")
                print("-" * 50)
                
                # Send follow-up message with same context
                followup_request = SendStreamingMessageRequest(
                    id=str(uuid4()),
                    params=MessageSendParams(
                        message={
                            "role": "user",
                            "parts": [{"kind": "text", "text": user_input}],
                            "messageId": uuid4().hex,
                            "contextId": context_id,
                            "taskId": task_id,
                        }
                    ),
                )
                
                # Stream the follow-up response
                stream_response = a2a_client.send_message_streaming(followup_request)
                async for chunk in stream_response:
                    if hasattr(chunk, "root") and hasattr(chunk.root, "result"):
                        result = chunk.root.result
                        
                        if isinstance(result, TaskStatusUpdateEvent):
                            if hasattr(result, "status") and hasattr(result.status, "state"):
                                state = result.status.state
                                
                                if state == TaskState.completed:
                                    print("\n✅ Task completed after user input!")
                                    if hasattr(result.status, "message") and result.status.message:
                                        if hasattr(result.status.message, "parts"):
                                            for part in result.status.message.parts:
                                                if hasattr(part, "root") and hasattr(part.root, "text"):
                                                    print(f"Final result: {part.root.text}")
                                    break
            
        except Exception as e:
            logger.error(f"Error testing butler: {e}")
            raise


if __name__ == "__main__":
    asyncio.run(test_butler_input_required())
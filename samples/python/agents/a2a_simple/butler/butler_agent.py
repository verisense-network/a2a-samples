"""Butler Agent - Orchestrates multiple A2A agents"""

import asyncio
import json
import logging
from typing import List, Dict, Any, AsyncIterable, Optional, Tuple
from uuid import uuid4
import httpx
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from pydantic import BaseModel, Field
from .agent_call import agent_call
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    MessageSendParams,
    SendMessageRequest,
    SendStreamingMessageRequest,
    TaskStatusUpdateEvent,
    TaskArtifactUpdateEvent,
    TaskState,
    Artifact,
    Task,
)

from app.agent import PromptBasedAgent, ResponseFormat
from app.rich_logging_config import (
    get_rich_logger,
    log_info,
    log_error,
    log_warning,
    log_success,
    log_debug,
)

# Get logger with rich formatting
logger = get_rich_logger(__name__)


class CallAgentError(Exception):
    """Error calling an agent"""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class QueryAgentsError(Exception):
    """Error querying available agents"""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class AgentInfo(BaseModel):
    """Information about an available A2A agent"""

    key: str
    name: str
    url: str
    description: str = ""


class TaskStep(BaseModel):
    """A single step in the task execution plan"""

    agent_key: str
    agent_name: str
    task_description: str
    depends_on: List[int] = Field(default_factory=list)


class ExecutionPlan(BaseModel):
    """The complete execution plan for a task"""

    status: str = "planning"
    original_query: str
    steps: List[TaskStep]
    final_summary: str = ""


class ButlerAgent(PromptBasedAgent):
    """Butler agent that orchestrates multiple A2A agents"""

    BUTLER_PROMPT = """You are a Butler Agent, an intelligent orchestrator that coordinates multiple specialized AI agents to complete complex tasks.

Your primary responsibilities:
1. Analyze user requests and break them down into subtasks
2. Query available A2A agents to understand their capabilities
3. Create an execution plan that assigns subtasks to appropriate agents
4. Execute the plan by calling agents in the correct sequence
5. Pass relevant context and results between agents
6. Synthesize all agent responses into a coherent final answer

When creating execution plans:
- Consider agent specializations and choose the most appropriate agent for each subtask
- Ensure logical task sequencing where dependent tasks come after their prerequisites
- Include all necessary context when calling each agent
- Track and report progress throughout execution

You have access to query all available A2A agents and coordinate their responses to provide comprehensive solutions."""

    def __init__(self, rpc_url: str = "https://rpc.beta.verisense.network"):
        super().__init__(system_prompt=self.BUTLER_PROMPT, use_tools=None)
        self.rpc_url = rpc_url
        self.available_agents: List[AgentInfo] = []

    async def query_available_agents(self) -> List[AgentInfo]:
        """Query the RPC endpoint to get list of available A2A agents"""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    self.rpc_url,
                    json={
                        "id": 1,
                        "jsonrpc": "2.0",
                        "method": "a2a_list",
                        "params": [],
                    },
                    headers={"Content-Type": "application/json"},
                )

                if response.status_code == 200:
                    data = response.json()
                    if "result" in data:
                        agents = []
                        # The result is a list of [key, agent_data] pairs
                        for item in data["result"]:
                            if isinstance(item, list) and len(item) == 2:
                                key, agent_data = item
                                if isinstance(agent_data, dict):
                                    agents.append(
                                        AgentInfo(
                                            key=key,
                                            name=agent_data.get("name", ""),
                                            url=agent_data.get("url", ""),
                                            description=agent_data.get(
                                                "description", ""
                                            ),
                                        )
                                    )
                        return agents
                else:
                    logger.error(f"Failed to query agents: {response.status_code}")

            except Exception as e:
                logger.error(f"Error querying available agents: {e}")
                raise QueryAgentsError(f"Error querying available agents: {e}")

        return []

    async def call_agent(
        self,
        agent_info: AgentInfo,
        query: str,
        context: str = "",
        task_id: str = None,
        context_id: str = None,
    ) -> Tuple[str, List[Artifact], Optional[Dict]]:
        """Call a specific A2A agent using the A2A protocol with streaming"""
        # Increase timeout for code generation agents
        timeout = httpx.Timeout(120.0, connect=30.0)
        # Shorter timeout for review agents
        # if "review" in query.lower() or agent_info.name == "Software Engineer":
        #     timeout = httpx.Timeout(30.0, connect=10.0)

        async with httpx.AsyncClient(timeout=timeout) as httpx_client:
            try:
                # Step 1: Get agent card
                resolver = A2ACardResolver(
                    httpx_client=httpx_client,
                    base_url=agent_info.url,
                )

                try:
                    agent_card = await resolver.get_agent_card()
                except Exception as e:
                    raise CallAgentError(
                        f"Failed to get agent card for {agent_info.name}, falling back to direct call: {e}"
                    )

                # Step 2: Initialize A2A client
                client = A2AClient(
                    httpx_client=httpx_client, agent_card=agent_card, url=agent_info.url
                )

                # Prepare the message with context
                full_query = query
                if context:
                    full_query = (
                        f"Context from previous agents:\n{context}\n\nTask: {query}"
                    )

                # Step 3: Use streaming to monitor task progress
                streaming_request = SendStreamingMessageRequest(
                    id=str(uuid4()),
                    params=MessageSendParams(
                        message={
                            "role": "user",
                            "parts": [{"kind": "text", "text": full_query}],
                            "messageId": uuid4().hex,
                        }
                    ),
                )

                # Stream the response and wait for completion
                final_response = ""
                task_completed = False
                task_requires_input = False
                task_id_from_stream = None
                all_chunks = []  # Store all chunks for debugging
                artifact_content = []  # Store artifact content as we receive it

                stream_response = client.send_message_streaming(streaming_request)
                artifact_list = []
                try:
                    async for chunk in stream_response:
                        logger.debug(f"Streaming chunk type: {type(chunk)}")

                        # Store raw chunk for potential fallback processing
                        if hasattr(chunk, "model_dump"):
                            chunk_data = chunk.model_dump()
                            all_chunks.append(chunk_data)

                            # Log specific chunk types at debug level
                            if isinstance(chunk_data, dict) and "result" in chunk_data:
                                result_kind = chunk_data.get("result", {}).get(
                                    "kind", "unknown"
                                )
                                logger.debug(f"Received chunk kind: {result_kind}")
                                if result_kind == "artifact-update":
                                    logger.debug(
                                        f"Artifact chunk: {chunk_data.get('result', {}).get('artifact', {})}"
                                    )

                        # Check if this is a task status update
                        if hasattr(chunk, "root") and hasattr(chunk.root, "result"):
                            result = chunk.root.result

                            # Handle TaskStatusUpdateEvent
                            if isinstance(result, TaskStatusUpdateEvent):
                                if hasattr(result, "taskId"):
                                    task_id_from_stream = result.taskId

                                if hasattr(result, "status") and hasattr(
                                    result.status, "state"
                                ):
                                    task_state = result.status.state
                                    logger.debug(f"Task state update: {task_state}")

                                    # Check if task requires input
                                    if task_state == TaskState.input_required:
                                        task_requires_input = True
                                        logger.info("Task requires user input")
                                        # Get the message from the status update if available
                                        if (
                                            hasattr(result.status, "message")
                                            and result.status.message
                                        ):
                                            if hasattr(result.status.message, "parts"):
                                                for part in result.status.message.parts:
                                                    if hasattr(
                                                        part, "root"
                                                    ) and hasattr(part.root, "text"):
                                                        text = part.root.text
                                                        if text:
                                                            final_response += text
                                    elif task_state == TaskState.working:
                                        if (
                                            hasattr(result.status, "message")
                                            and result.status.message
                                        ):
                                            if hasattr(result.status.message, "parts"):
                                                for part in result.status.message.parts:
                                                    if hasattr(
                                                        part, "root"
                                                    ) and hasattr(part.root, "text"):
                                                        text = part.root.text
                                                        if text:
                                                            final_response += text
                                    # Check if task is completed
                                    elif task_state == TaskState.completed:
                                        task_completed = True
                                        # Get the message from the status update if available
                                        if (
                                            hasattr(result.status, "message")
                                            and result.status.message
                                        ):
                                            if hasattr(result.status.message, "parts"):
                                                for part in result.status.message.parts:
                                                    if hasattr(
                                                        part, "root"
                                                    ) and hasattr(part.root, "text"):
                                                        text = part.root.text
                                                        if (
                                                            text
                                                            and not text.startswith("⏲️")
                                                        ):
                                                            final_response += text

                            # Handle TaskArtifactUpdateEvent
                            elif isinstance(result, TaskArtifactUpdateEvent):
                                logger.debug(f"Artifact update event: {result}")
                                if (
                                    hasattr(result, "artifact")
                                    and result.artifact
                                    and isinstance(result.artifact, Artifact)
                                ):
                                    artifact_list.append(result.artifact)
                                    # Get artifact name if available
                                    artifact_name = getattr(
                                        result.artifact,
                                        "name",
                                        getattr(result.artifact, "artifactId", "code"),
                                    )
                                    if hasattr(result.artifact, "parts"):
                                        for part in result.artifact.parts:
                                            try:
                                                # Handle different part structures
                                                if hasattr(part, "root"):
                                                    root = part.root

                                                    # Handle text parts
                                                    if hasattr(root, "text"):
                                                        text = root.text
                                                        if text:
                                                            # Determine file extension for formatting
                                                            file_ext = ""
                                                            if artifact_name:
                                                                if artifact_name.endswith(
                                                                    ".py"
                                                                ):
                                                                    file_ext = "python"
                                                                elif artifact_name.endswith(
                                                                    ".js"
                                                                ):
                                                                    file_ext = (
                                                                        "javascript"
                                                                    )
                                                                elif artifact_name.endswith(
                                                                    ".ts"
                                                                ):
                                                                    file_ext = (
                                                                        "typescript"
                                                                    )
                                                                elif artifact_name.endswith(
                                                                    ".java"
                                                                ):
                                                                    file_ext = "java"
                                                                elif artifact_name.endswith(
                                                                    ".cpp"
                                                                ) or artifact_name.endswith(
                                                                    ".cc"
                                                                ):
                                                                    file_ext = "cpp"
                                                                elif artifact_name.endswith(
                                                                    ".c"
                                                                ):
                                                                    file_ext = "c"
                                                                elif artifact_name.endswith(
                                                                    ".md"
                                                                ):
                                                                    file_ext = (
                                                                        "markdown"
                                                                    )
                                                                elif artifact_name.endswith(
                                                                    ".html"
                                                                ):
                                                                    file_ext = "html"
                                                                elif artifact_name.endswith(
                                                                    ".css"
                                                                ):
                                                                    file_ext = "css"
                                                                elif artifact_name.endswith(
                                                                    ".json"
                                                                ):
                                                                    file_ext = "json"
                                                                elif artifact_name.endswith(
                                                                    ".yaml"
                                                                ) or artifact_name.endswith(
                                                                    ".yml"
                                                                ):
                                                                    file_ext = "yaml"
                                                                elif artifact_name.endswith(
                                                                    ".xml"
                                                                ):
                                                                    file_ext = "xml"
                                                                elif artifact_name.endswith(
                                                                    ".sh"
                                                                ):
                                                                    file_ext = "bash"
                                                                elif artifact_name.endswith(
                                                                    ".sql"
                                                                ):
                                                                    file_ext = "sql"

                                                            # Format content with appropriate code block
                                                            if file_ext:
                                                                artifact_content.append(
                                                                    f"```{file_ext}\n# {artifact_name}\n{text}\n```"
                                                                )
                                                            else:
                                                                # Fallback for unknown file types
                                                                artifact_content.append(
                                                                    text
                                                                )

                                                    # Handle file parts with kind property
                                                    elif (
                                                        hasattr(root, "kind")
                                                        and root.kind == "file"
                                                    ):
                                                        file_name = getattr(
                                                            root,
                                                            "fileName",
                                                            artifact_name
                                                            or "unnamed_file",
                                                        )
                                                        file_content = getattr(
                                                            root, "text", None
                                                        )

                                                        if file_content:
                                                            # Determine language from file extension
                                                            if file_name.endswith(
                                                                ".py"
                                                            ):
                                                                lang = "python"
                                                            elif file_name.endswith(
                                                                ".js"
                                                            ):
                                                                lang = "javascript"
                                                            elif file_name.endswith(
                                                                ".ts"
                                                            ):
                                                                lang = "typescript"
                                                            elif file_name.endswith(
                                                                ".java"
                                                            ):
                                                                lang = "java"
                                                            elif file_name.endswith(
                                                                (".cpp", ".cc", ".cxx")
                                                            ):
                                                                lang = "cpp"
                                                            elif file_name.endswith(
                                                                ".c"
                                                            ):
                                                                lang = "c"
                                                            elif file_name.endswith(
                                                                ".go"
                                                            ):
                                                                lang = "go"
                                                            elif file_name.endswith(
                                                                ".rs"
                                                            ):
                                                                lang = "rust"
                                                            elif file_name.endswith(
                                                                ".php"
                                                            ):
                                                                lang = "php"
                                                            elif file_name.endswith(
                                                                ".rb"
                                                            ):
                                                                lang = "ruby"
                                                            elif file_name.endswith(
                                                                ".swift"
                                                            ):
                                                                lang = "swift"
                                                            elif file_name.endswith(
                                                                ".kt"
                                                            ):
                                                                lang = "kotlin"
                                                            elif file_name.endswith(
                                                                ".scala"
                                                            ):
                                                                lang = "scala"
                                                            elif file_name.endswith(
                                                                ".r"
                                                            ):
                                                                lang = "r"
                                                            elif file_name.endswith(
                                                                ".m"
                                                            ):
                                                                lang = "objective-c"
                                                            else:
                                                                lang = "text"

                                                            artifact_content.append(
                                                                f"```{lang}\n# {file_name}\n{file_content}\n```"
                                                            )

                                                    # Handle other root types that might contain text
                                                    elif hasattr(root, "content"):
                                                        content = root.content
                                                        if content:
                                                            artifact_content.append(
                                                                content
                                                            )

                                                # Handle parts without root property (direct text)
                                                elif hasattr(part, "text"):
                                                    text = part.text
                                                    if text:
                                                        artifact_content.append(text)

                                                # Handle parts with direct file properties
                                                elif hasattr(
                                                    part, "fileName"
                                                ) and hasattr(part, "content"):
                                                    file_name = part.fileName
                                                    file_content = part.content
                                                    if file_content:
                                                        # Simple language detection
                                                        if file_name.endswith(".py"):
                                                            lang = "python"
                                                        elif file_name.endswith(
                                                            (".js", ".jsx")
                                                        ):
                                                            lang = "javascript"
                                                        elif file_name.endswith(
                                                            (".ts", ".tsx")
                                                        ):
                                                            lang = "typescript"
                                                        else:
                                                            lang = "text"

                                                        artifact_content.append(
                                                            f"```{lang}\n# {file_name}\n{file_content}\n```"
                                                        )

                                                # Fallback: try to extract any string content
                                                else:
                                                    part_str = str(part)
                                                    if part_str and part_str != str(
                                                        part.__class__
                                                    ):
                                                        artifact_content.append(
                                                            part_str
                                                        )

                                            except Exception as part_error:
                                                logger.warning(
                                                    f"Error processing artifact part: {part_error}"
                                                )
                                                # Still try to get something useful from the part
                                                try:
                                                    part_str = str(part)
                                                    if (
                                                        part_str
                                                        and len(part_str) < 10000
                                                    ):  # Avoid huge dumps
                                                        artifact_content.append(
                                                            f"[Partial content]: {part_str}"
                                                        )
                                                except:
                                                    logger.warning(
                                                        "Could not extract any content from artifact part"
                                                    )

                                    # Handle artifacts without parts (direct content)
                                    elif hasattr(result.artifact, "content"):
                                        content = result.artifact.content
                                        if content:
                                            artifact_content.append(content)

                                    # Handle artifacts with direct text property
                                    elif hasattr(result.artifact, "text"):
                                        text = result.artifact.text
                                        if text:
                                            if (
                                                artifact_name
                                                and artifact_name.endswith(".py")
                                            ):
                                                artifact_content.append(
                                                    f"```python\n# {artifact_name}\n{text}\n```"
                                                )
                                            else:
                                                artifact_content.append(text)

                            # Handle other message types
                            elif hasattr(result, "message"):
                                message = result.message
                                if hasattr(message, "parts"):
                                    for part in message.parts:
                                        if hasattr(part, "root") and hasattr(
                                            part.root, "text"
                                        ):
                                            text = part.root.text
                                            # Keep track of non-progress messages
                                            if text and not text.startswith("⏲️"):
                                                final_response += text

                except Exception as stream_error:
                    logger.error(
                        f"Error during streaming, attempting to extract response from chunks: {stream_error}"
                    )
                    raise CallAgentError(
                        f"Error during streaming, attempting to extract response from chunks: {stream_error}"
                    )
                logger.info(f"Final response: {final_response}")
                # If we collected artifact content but no final response, use the artifacts
                if final_response == "" and artifact_content:
                    final_response = "\n\n".join(artifact_content)
                    # If we have artifacts, consider it completed
                    if not task_completed and artifact_content:
                        task_completed = True

                # Log final state
                logger.info(
                    f"Stream completed - task_completed: {task_completed}, task_requires_input: {task_requires_input}, has_final_response: {bool(final_response)}, artifact_count: {len(artifact_content)}"
                )

                # Handle input required case
                if task_requires_input:
                    return (
                        final_response or "User input required",
                        artifact_list,
                        {
                            "state": "input_required",
                            "task_id": task_id_from_stream,
                            "context_id": context_id
                            or streaming_request.params.message.get("contextId"),
                            "message": final_response,
                        },
                    )

                # Return the final response
                if task_completed and final_response != "":
                    return final_response, artifact_list, None
                elif task_completed:
                    return "Task completed but no response text found", [], None
                else:
                    # If we have artifacts but no completion status, still return them
                    if artifact_content:
                        logger.info("Returning artifacts despite no completion status")
                        return "\n\n".join(artifact_content), [], None
                    return "Task did not complete within timeout", [], None

            except Exception as e:
                logger.error(f"Error calling agent {agent_info.name}: {e}")
                raise CallAgentError(f"Error calling agent {agent_info.name}: {e}")

        return "No response from agent", [], None

    async def create_execution_plan(self, query: str) -> ExecutionPlan:
        """Create an execution plan for the given query"""

        # Get available agents
        self.available_agents = await self.query_available_agents()
        # Use LLM to create execution plan
        plan_prompt = f"""Given the user query and available agents, create an execution plan.

User Query: {query}

Available Agents:
{json.dumps([{"name": a.name, "key": a.key, "description": a.description} for a in self.available_agents], indent=2)}

Create a step-by-step plan to accomplish this task using the available agents.

IMPORTANT: 
- When the user's request is vague (e.g., "help me design something"), the first step should be to gather more information
- Don't make assumptions about what the user wants - let the specialized agents ask for clarification
- Include the original user query context when calling agents so they can ask follow-up questions if needed

Return the plan as a structured response."""

        messages = [("system", self.BUTLER_PROMPT), ("user", plan_prompt)]

        # Get structured plan from LLM
        structured_model = self.model.with_structured_output(ExecutionPlan)
        plan = structured_model.invoke(messages)
        plan.original_query = query

        return plan

    async def stream(
        self, query: str, context_id: str, conversation_parts=None
    ) -> AsyncIterable[dict[str, Any]]:
        """Stream the butler agent's execution process"""
        try:
            # Step 1: Query available agents
            yield {
                "is_task_complete": False,
                "require_user_input": False,
                "content": "⏲️ Querying available agents...\n",
            }

            agents = await self.query_available_agents()

            yield {
                "is_task_complete": False,
                "require_user_input": False,
                "content": f"⏲️ Found {len(agents)} available agents\n",
            }

            # Step 2: Create execution plan
            yield {
                "is_task_complete": False,
                "require_user_input": False,
                "content": "⏲️ Creating execution plan...\n",
            }

            plan = await self.create_execution_plan(query)

            yield {
                "is_task_complete": False,
                "require_user_input": False,
                "content": f"⏲️ Plan created with {len(plan.steps)} steps\n",
            }

            # Step 3: Execute plan
            results = []
            accumulated_context = ""
            artifacts = {}  # Store artifacts by agent name

            for i, step in enumerate(plan.steps):
                # Find the agent info
                agent_info = next((a for a in agents if a.key == step.agent_key), None)
                if not agent_info:
                    continue

                yield {
                    "is_task_complete": False,
                    "require_user_input": False,
                    "content": f"⏲️ Step {i+1}: @[{step.agent_name}] - {step.task_description}\n",
                }

                # Build context from dependent steps
                step_context = ""
                for dep_idx in step.depends_on:
                    if dep_idx < len(results):
                        step_context += f"\n{results[dep_idx]}"

                # If this is a code review step and context is empty, use accumulated context
                if (
                    "review" in step.task_description.lower()
                    and not step_context
                    and accumulated_context
                ):
                    step_context = accumulated_context

                # For the first step or when no context, include the original user query
                # This ensures agents get the full context of what the user wants
                if i == 0 or not step_context:
                    full_query = f"User request: {query}\n\nSpecific task: {step.task_description}"
                else:
                    full_query = step.task_description

                # Call the agent
                result, artifact_list, task_state = await self.call_agent(
                    agent_info, full_query, step_context
                )
                logger.info(
                    f"Agent {step.agent_name} returned: {result[:200]}, task_state: {task_state}"
                )
                # Check if agent requires user input
                if task_state and task_state.get("state") == "input_required":
                    # Butler handles this internally - add to results and continue
                    input_message = task_state.get(
                        "message", "Agent requires additional information"
                    )
                    result = f"[{step.agent_name} needs clarification: {input_message}]"
                    logger.info(
                        f"Agent {step.agent_name} requires input - butler will handle this in final summary"
                    )

                # Check if the agent returned an empty or generic response
                if result == "Task completed but no response text found":
                    # This might indicate the agent needs more information
                    logger.warning(
                        f"Agent {step.agent_name} returned empty response - may need clarification"
                    )
                    result = f"[{step.agent_name} completed the task but provided no specific response - this may indicate that more specific information is needed]"

                results.append(result)
                for artifact in artifact_list:
                    yield {
                        "is_task_complete": False,
                        "require_user_input": False,
                        "artifact": artifact,
                    }

                # Store artifact if it looks like code
                if "```" in result or result.strip().startswith("#"):
                    artifacts[step.agent_name] = result

                accumulated_context += f"\n\n{step.agent_name}: {result}"

            # Step 4: Synthesize final answer
            yield {
                "is_task_complete": False,
                "require_user_input": False,
                "content": "⏲️ Synthesizing final answer...\n",
            }

            # Use LLM to create final summary
            summary_prompt = f"""Based on the execution results, provide a comprehensive answer to the original query.

Original Query: {query}

Execution Results:
{accumulated_context}

Synthesize all the information into a clear, coherent response. If code was generated, include it in your response.

IMPORTANT: 
- If any agent indicated they need more specific information (e.g., empty responses, requests for clarification, or "needs clarification" messages), 
  you MUST provide a helpful response that guides the user on what specific information is needed.
- Do NOT ask for input or wait for more information - provide a complete response with clear guidance.
- Example: If an Industrial Designer needs to know what to design, suggest specific options or ask clarifying questions in your final response."""

            messages = [("system", self.BUTLER_PROMPT), ("user", summary_prompt)]

            response = self.model.invoke(messages)

            # Always complete the task - never require user input
            yield {
                "is_task_complete": True,
                "require_user_input": False,
                "content": f"✅ {response.content}",
            }

        except Exception as e:
            logger.error(f"Error in butler execution: {e}", exc_info=True)
            yield {
                "is_task_complete": True,
                "require_user_input": False,
                "error": True,
                "content": f"❌ Error during execution: {str(e)}",
            }

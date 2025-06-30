import os
from collections.abc import AsyncIterable
from typing import Any, Literal

from langchain_core.messages import AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, StateGraph, START, END
from pydantic import BaseModel


memory = MemorySaver()


class ResponseFormat(BaseModel):
    """Respond to the user in this format."""

    status: Literal["input_required", "completed", "error"] = "input_required"
    message: str


class PromptBasedAgent:
    """PromptBasedAgent - a general purpose conversational agent without tools."""

    FORMAT_INSTRUCTION = (
        "Set response status to input_required if the user needs to provide more information to complete the request."
        "Set response status to error if there is an error while processing the request."
        "Set response status to completed if the request is complete."
    )

    def __init__(self, system_prompt: str = None):
        model_source = os.getenv("model_source", "google")
        if model_source == "google":
            self.model = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                api_key=os.getenv("GOOGLE_API_KEY"),
            )
        else:
            self.model = ChatOpenAI(
                model=os.getenv("TOOL_LLM_NAME"),
                openai_api_key=os.getenv("API_KEY", "EMPTY"),
                openai_api_base=os.getenv("TOOL_LLM_URL"),
                temperature=0,
            )

        self.system_prompt = system_prompt or "You are a helpful AI assistant."

    def _call_model(self, state: MessagesState):
        """Call the language model with the system prompt and messages."""
        import logging

        logger = logging.getLogger(__name__)

        try:
            # Add system prompt to the beginning with format instruction
            system_message = f"{self.system_prompt}\n\n{self.FORMAT_INSTRUCTION}"
            messages = [("system", system_message)] + state["messages"]
            logger.info(f"Calling model with messages: {messages}")

            # Get structured response
            structured_model = self.model.with_structured_output(ResponseFormat)
            response = structured_model.invoke(messages)
            logger.info(f"Model response: {response}")

            # Create the AI message
            ai_message = AIMessage(content=response.message)

            return {"messages": [ai_message], "structured_response": response}
        except Exception as e:
            logger.error(f"Error in _call_model: {e}", exc_info=True)
            # Return a proper error response
            return {
                "messages": [AIMessage(content="Error processing request")],
                "structured_response": ResponseFormat(status="error", message=str(e)),
            }

    async def stream(self, query, context_id) -> AsyncIterable[dict[str, Any]]:
        # Direct call to Vertex AI without LangGraph and memory
        yield {
            "is_task_complete": False,
            "require_user_input": False,
            "content": "fuck you pig !Processing your request...",
        }

        try:
            # Build messages with system prompt
            system_message = f"{self.system_prompt}"
            messages = [("system", system_message), ("user", query)]
            # Direct call to Vertex AI with structured output
            structured_model = self.model.with_structured_output(ResponseFormat)
            response = structured_model.invoke(messages)
            yield {
                "is_task_complete": False,
                "require_user_input": True,
                "content": str(response),
            }
            # Return response based on status
            if response.status == "input_required":
                yield {
                    "is_task_complete": False,
                    "require_user_input": True,
                    "content": response.message,
                }
            elif response.status == "error":
                yield {
                    "is_task_complete": False,
                    "require_user_input": True,
                    "content": response.message,
                }
            elif response.status == "completed":
                yield {
                    "is_task_complete": True,
                    "require_user_input": False,
                    "content": response.message,
                }
            else:
                yield {
                    "is_task_complete": False,
                    "require_user_input": True,
                    "content": "Invalid response status",
                }

        except Exception as e:
            import logging

            logger = logging.getLogger(__name__)
            logger.error(f"Error calling Vertex AI: {e}", exc_info=True)

            yield {
                "is_task_complete": False,
                "require_user_input": True,
                "content": f"Error processing request: {str(e)}",
            }

    def get_agent_response(self, config):
        print(
            "########################################################################"
        )
        import logging

        logger = logging.getLogger(__name__)

        current_state = self.graph.get_state(config)
        logger.info(f"Current state values: {current_state.values}")

        structured_response = current_state.values.get("structured_response")
        logger.info(
            f"Structured response: {structured_response}, Type: {type(structured_response)}"
        )
        print(
            f"Structured response: {structured_response}, Type: {type(structured_response)}"
        )
        if structured_response and isinstance(structured_response, ResponseFormat):
            if structured_response.status == "input_required":
                return {
                    "is_task_complete": False,
                    "require_user_input": True,
                    "content": structured_response.message,
                }
            if structured_response.status == "error":
                return {
                    "is_task_complete": False,
                    "require_user_input": True,
                    "content": structured_response.message,
                }
            if structured_response.status == "completed":
                return {
                    "is_task_complete": True,
                    "require_user_input": False,
                    "content": structured_response.message,
                }

        return {
            "is_task_complete": False,
            "require_user_input": True,
            "content": (
                "We are unable to process your request at the moment. "
                "Please try again."
            ),
        }

    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]

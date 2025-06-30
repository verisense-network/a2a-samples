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

    status: Literal['input_required', 'completed', 'error'] = 'input_required'
    message: str


class PromptBasedAgent:
    """PromptBasedAgent - a general purpose conversational agent without tools."""

    FORMAT_INSTRUCTION = (
        'Set response status to input_required if the user needs to provide more information to complete the request.'
        'Set response status to error if there is an error while processing the request.'
        'Set response status to completed if the request is complete.'
    )

    def __init__(self, system_prompt: str = None):
        model_source = os.getenv('model_source', 'google')
        if model_source == 'google':
            self.model = ChatGoogleGenerativeAI(model='gemini-2.0-flash')
        else:
            self.model = ChatOpenAI(
                model=os.getenv('TOOL_LLM_NAME'),
                openai_api_key=os.getenv('API_KEY', 'EMPTY'),
                openai_api_base=os.getenv('TOOL_LLM_URL'),
                temperature=0,
            )
        
        self.system_prompt = system_prompt or "You are a helpful AI assistant."
        
        # Create a simple graph without tools
        workflow = StateGraph(MessagesState)
        
        # Add the agent node
        workflow.add_node("agent", self._call_model)
        
        # Add edge from START to agent
        workflow.add_edge(START, "agent")
        
        # Add edge from agent to END
        workflow.add_edge("agent", END)
        
        # Compile the graph
        self.graph = workflow.compile(checkpointer=memory)

    def _call_model(self, state: MessagesState):
        """Call the language model with the system prompt and messages."""
        # Add system prompt to the beginning
        messages = [("system", self.system_prompt)] + state["messages"]
        
        # Get structured response
        structured_model = self.model.with_structured_output(ResponseFormat)
        response = structured_model.invoke(messages)
        
        # Create the AI message
        ai_message = AIMessage(content=response.message)
        
        return {
            "messages": [ai_message],
            "structured_response": response
        }

    async def stream(self, query, context_id) -> AsyncIterable[dict[str, Any]]:
        inputs = {'messages': [('user', query)]}
        config = {'configurable': {'thread_id': context_id}}

        # For simple prompt-based agents, we don't have intermediate steps
        # Just process the query and return the response
        yield {
            'is_task_complete': False,
            'require_user_input': False,
            'content': 'Processing your request...',
        }

        # Run the graph
        _ = self.graph.invoke(inputs, config)
        
        # Get the final response
        yield self.get_agent_response(config)

    def get_agent_response(self, config):
        current_state = self.graph.get_state(config)
        structured_response = current_state.values.get('structured_response')
        if structured_response and isinstance(
            structured_response, ResponseFormat
        ):
            if structured_response.status == 'input_required':
                return {
                    'is_task_complete': False,
                    'require_user_input': True,
                    'content': structured_response.message,
                }
            if structured_response.status == 'error':
                return {
                    'is_task_complete': False,
                    'require_user_input': True,
                    'content': structured_response.message,
                }
            if structured_response.status == 'completed':
                return {
                    'is_task_complete': True,
                    'require_user_input': False,
                    'content': structured_response.message,
                }

        return {
            'is_task_complete': False,
            'require_user_input': True,
            'content': (
                'We are unable to process your request at the moment. '
                'Please try again.'
            ),
        }

    SUPPORTED_CONTENT_TYPES = ['text', 'text/plain']
import json
import logging
import os
import sys
from pathlib import Path

import click
import httpx
import uvicorn

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryPushNotifier, InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)
from dotenv import load_dotenv

from app.agent import PromptBasedAgent
from app.agent_executor import PromptAgentExecutor


load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MissingAPIKeyError(Exception):
    """Exception for missing API key."""


def load_agent_config(agent_index: int = 0):
    """Load agent configuration from agent-prompts.json."""
    try:
        with open("agent-prompts.json", "r") as f:
            agents = json.load(f)

        if 0 <= agent_index < len(agents):
            return agents[agent_index]
        else:
            logger.warning(
                f"Agent index {agent_index} out of range, using default agent"
            )
            return None
    except FileNotFoundError:
        logger.warning("agent-prompts.json not found, using default agent")
        return None
    except json.JSONDecodeError:
        logger.error("Error parsing agent-prompts.json")
        return None


@click.command()
@click.option("--host", "host", default="localhost")
@click.option("--port", "port", default=10000)
@click.option(
    "--agent-index",
    "agent_index",
    default=-1,
    help="Index of agent in agent-prompts.json (0-99). If not specified, runs a generic agent.",
)
def main(host, port, agent_index):
    """Starts the Agent server."""
    try:
        if os.getenv("model_source", "google") == "google":
            if not os.getenv("GOOGLE_API_KEY"):
                raise MissingAPIKeyError("GOOGLE_API_KEY environment variable not set.")
        else:
            if not os.getenv("TOOL_LLM_URL"):
                raise MissingAPIKeyError("TOOL_LLM_URL environment variable not set.")
            if not os.getenv("TOOL_LLM_NAME"):
                raise MissingAPIKeyError(
                    "TOOL_LLM_NAME environment not variable not set."
                )

        # Load agent configuration if specified
        agent_config = None
        agent_name = "AI Assistant"
        agent_prompt = "You are a helpful AI assistant."

        if agent_index >= 0:
            agent_config = load_agent_config(agent_index)
            if agent_config:
                agent_name = agent_config.get("name", agent_name)
                agent_prompt = agent_config.get("prompt", agent_prompt)
                logger.info(f"Loaded agent: {agent_name}")

        capabilities = AgentCapabilities(streaming=True, pushNotifications=True)
        skill = AgentSkill(
            id=f'{agent_name.lower().replace(" ", "_")}_skill',
            name=f"{agent_name} Assistant",
            description=f"AI assistant specialized as {agent_name}",
            tags=[agent_name.lower(), "assistant", "ai"],
            examples=[f"Help me with {agent_name.lower()} related tasks"],
        )
        agent_card = AgentCard(
            name=agent_name,
            description=f"{agent_name} - AI Assistant",
            url=f"http://34.57.6.105/p{port}",
            version="1.0.0",
            defaultInputModes=PromptBasedAgent.SUPPORTED_CONTENT_TYPES,
            defaultOutputModes=PromptBasedAgent.SUPPORTED_CONTENT_TYPES,
            capabilities=capabilities,
            skills=[skill],
        )

        # --8<-- [start:DefaultRequestHandler]
        httpx_client = httpx.AsyncClient()
        request_handler = DefaultRequestHandler(
            agent_executor=PromptAgentExecutor(system_prompt=agent_prompt),
            task_store=InMemoryTaskStore(),
            push_notifier=InMemoryPushNotifier(httpx_client),
        )
        server = A2AStarletteApplication(
            agent_card=agent_card, http_handler=request_handler
        )

        uvicorn.run(server.build(), host=host, port=port)
        # --8<-- [end:DefaultRequestHandler]

    except MissingAPIKeyError as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An error occurred during server startup: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

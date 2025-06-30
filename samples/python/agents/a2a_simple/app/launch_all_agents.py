#!/usr/bin/env python3
"""Launch all 100 agents on ports 10001-10100."""

import asyncio
import json
import logging
import os
import signal
import sys
from pathlib import Path

import click
import uvicorn
from dotenv import load_dotenv

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryPushNotifier, InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
import httpx

from app.agent_executor import PromptAgentExecutor
from app.agent import PromptBasedAgent

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentProcess:
    """Represents a single agent process."""
    
    def __init__(self, agent_data: dict, port: int, host: str = 'localhost'):
        self.name = agent_data['name']
        self.prompt = agent_data['prompt']
        self.port = port
        self.host = host
        self.server = None
        self.process = None
        
    async def start(self):
        """Start the agent server."""
        try:
            # Create agent-specific capabilities
            capabilities = AgentCapabilities(streaming=True, pushNotifications=True)
            
            # Create agent-specific skill
            skill = AgentSkill(
                id=f'{self.name.lower().replace(" ", "_")}_skill',
                name=f'{self.name} Assistant',
                description=f'AI assistant specialized as {self.name}',
                tags=[self.name.lower(), 'assistant', 'ai'],
                examples=[f'Help me with {self.name.lower()} tasks'],
            )
            
            # Create agent card with specific information
            agent_card = AgentCard(
                name=self.name,
                description=f'{self.name} - AI Assistant',
                url=f'http://{self.host}:{self.port}/',
                version='1.0.0',
                defaultInputModes=PromptBasedAgent.SUPPORTED_CONTENT_TYPES,
                defaultOutputModes=PromptBasedAgent.SUPPORTED_CONTENT_TYPES,
                capabilities=capabilities,
                skills=[skill],
            )
            
            # Create request handler with agent-specific prompt
            httpx_client = httpx.AsyncClient()
            request_handler = DefaultRequestHandler(
                agent_executor=PromptAgentExecutor(system_prompt=self.prompt),
                task_store=InMemoryTaskStore(),
                push_notifier=InMemoryPushNotifier(httpx_client),
            )
            
            # Create server
            self.server = A2AStarletteApplication(
                agent_card=agent_card, http_handler=request_handler
            )
            
            # Run server
            config = uvicorn.Config(
                self.server.build(),
                host=self.host,
                port=self.port,
                log_level="info",
                access_log=False
            )
            server = uvicorn.Server(config)
            
            logger.info(f"Starting {self.name} on port {self.port}")
            await server.serve()
            
        except Exception as e:
            logger.error(f"Error starting {self.name} on port {self.port}: {e}")
            raise


class MultiAgentLauncher:
    """Manages multiple agent processes."""
    
    def __init__(self, agents_file: str = 'agent-prompts.json'):
        self.agents_file = Path(agents_file)
        self.agents = []
        self.tasks = []
        
    def load_agents(self):
        """Load agent configurations from JSON file."""
        with open(self.agents_file, 'r') as f:
            agents_data = json.load(f)
        
        # Create agent processes for ports 10001-10100
        for i, agent_data in enumerate(agents_data[:100]):  # Ensure we only use 100 agents
            port = 10001 + i
            agent = AgentProcess(agent_data, port)
            self.agents.append(agent)
            
    async def start_all(self):
        """Start all agent servers concurrently."""
        logger.info(f"Starting {len(self.agents)} agents...")
        
        # Create tasks for all agents
        for agent in self.agents:
            task = asyncio.create_task(agent.start())
            self.tasks.append(task)
        
        # Wait for all tasks
        try:
            await asyncio.gather(*self.tasks)
        except KeyboardInterrupt:
            logger.info("Shutting down all agents...")
            await self.shutdown()
    
    async def shutdown(self):
        """Gracefully shutdown all agents."""
        for task in self.tasks:
            task.cancel()
        
        await asyncio.gather(*self.tasks, return_exceptions=True)
        logger.info("All agents shut down.")


def check_api_keys():
    """Check if required API keys are set."""
    if os.getenv('model_source', 'google') == 'google':
        if not os.getenv('GOOGLE_API_KEY'):
            logger.error('GOOGLE_API_KEY environment variable not set.')
            return False
    else:
        if not os.getenv('TOOL_LLM_URL'):
            logger.error('TOOL_LLM_URL environment variable not set.')
            return False
        if not os.getenv('TOOL_LLM_NAME'):
            logger.error('TOOL_LLM_NAME environment variable not set.')
            return False
    return True


@click.command()
@click.option('--host', default='localhost', help='Host to bind the agents to')
@click.option('--agents-file', default='agent-prompts.json', help='Path to agents JSON file')
def main(host: str, agents_file: str):
    """Launch all 100 agents on ports 10001-10100."""
    
    # Check API keys
    if not check_api_keys():
        sys.exit(1)
    
    # Create launcher
    launcher = MultiAgentLauncher(agents_file)
    
    try:
        # Load agent configurations
        launcher.load_agents()
        logger.info(f"Loaded {len(launcher.agents)} agent configurations")
        
        # Start all agents
        asyncio.run(launcher.start_all())
        
    except FileNotFoundError:
        logger.error(f"Agent prompts file not found: {agents_file}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
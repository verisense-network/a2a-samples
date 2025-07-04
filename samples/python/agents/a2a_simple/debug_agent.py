#!/usr/bin/env python3
"""
Debug script to test the agent directly without the A2A server layer.
This helps isolate whether the issue is in the agent logic or the server integration.
"""

import asyncio
import logging
import os
from dotenv import load_dotenv

# Set up logging to see all debug messages
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Load environment variables
load_dotenv()

# Import after loading env vars
from app.agent import PromptBasedAgent


async def test_agent():
    """Test the agent with a simple 'hello' message."""
    print("=== Testing PromptBasedAgent ===")
    print(f"Model source: {os.getenv('model_source', 'google')}")
    print(f"Google API Key present: {'GOOGLE_API_KEY' in os.environ}")
    
    # Create agent with default prompt
    agent = PromptBasedAgent()
    print(f"System prompt: {agent.system_prompt}")
    
    # Test the agent
    print("\n--- Sending 'hello' message ---")
    
    responses = []
    async for response in agent.stream("hello", "test-context-123"):
        print(f"Response: {response}")
        responses.append(response)
    
    print(f"\n--- Total responses: {len(responses)} ---")
    
    # Also test with a more specific query
    print("\n--- Sending 'What is 2+2?' message ---")
    
    responses2 = []
    async for response in agent.stream("What is 2+2?", "test-context-456"):
        print(f"Response: {response}")
        responses2.append(response)
    
    print(f"\n--- Total responses: {len(responses2)} ---")


if __name__ == "__main__":
    asyncio.run(test_agent())
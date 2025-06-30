#!/usr/bin/env python3
"""Test client for multiple agents running on different ports."""

import asyncio
import json
import logging
import random
from typing import List
from uuid import uuid4

import click
import httpx
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import MessageSendParams, SendMessageRequest


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_agent(port: int, agent_name: str, test_message: str) -> dict:
    """Test a single agent."""
    base_url = f'http://localhost:{port}'
    
    try:
        async with httpx.AsyncClient() as httpx_client:
            # Initialize resolver
            resolver = A2ACardResolver(
                httpx_client=httpx_client,
                base_url=base_url,
            )
            
            # Get agent card
            agent_card = await resolver.get_agent_card()
            logger.info(f"Connected to {agent_card.name} on port {port}")
            
            # Initialize client
            client = A2AClient(
                httpx_client=httpx_client,
                a2a_card=agent_card,
                base_url=base_url,
            )
            
            # Send message
            request = SendMessageRequest(
                id=str(uuid4()),
                jsonrpc='2.0',
                method='message/send',
                params=MessageSendParams(
                    message={
                        'kind': 'message',
                        'messageId': str(uuid4()),
                        'role': 'user',
                        'parts': [
                            {
                                'kind': 'text',
                                'text': test_message,
                            }
                        ],
                    }
                ),
            )
            
            response = await client.send_message(request)
            
            # Extract response content
            if response.result and hasattr(response.result, 'artifacts'):
                artifacts = response.result.artifacts
                if artifacts:
                    content = artifacts[0].parts[0].root.text
                    return {
                        'agent': agent_name,
                        'port': port,
                        'status': 'success',
                        'response': content
                    }
            
            return {
                'agent': agent_name,
                'port': port,
                'status': 'error',
                'response': 'No response content'
            }
            
    except Exception as e:
        return {
            'agent': agent_name,
            'port': port,
            'status': 'error',
            'response': str(e)
        }


async def test_multiple_agents(start_port: int, end_port: int, num_tests: int = 5):
    """Test multiple agents with random selection."""
    # Load agent configurations
    try:
        with open('agent-prompts.json', 'r') as f:
            agents = json.load(f)
    except FileNotFoundError:
        logger.error("agent-prompts.json not found")
        return
    
    # Test messages for different agent types
    test_messages = [
        "What are your main responsibilities?",
        "How can you help me today?",
        "What is your area of expertise?",
        "Can you provide an example of what you do?",
        "What makes you unique as an AI assistant?"
    ]
    
    # Randomly select agents to test
    num_agents = min(end_port - start_port + 1, len(agents))
    test_indices = random.sample(range(num_agents), min(num_tests, num_agents))
    
    tasks = []
    for idx in test_indices:
        port = start_port + idx
        agent_name = agents[idx]['name']
        test_message = random.choice(test_messages)
        
        logger.info(f"Queuing test for {agent_name} on port {port}")
        task = test_agent(port, agent_name, test_message)
        tasks.append(task)
    
    # Run tests concurrently
    results = await asyncio.gather(*tasks)
    
    # Display results
    print("\n" + "="*80)
    print("TEST RESULTS")
    print("="*80)
    
    for result in results:
        print(f"\nAgent: {result['agent']} (Port {result['port']})")
        print(f"Status: {result['status']}")
        print(f"Response: {result['response'][:200]}..." if len(result['response']) > 200 else f"Response: {result['response']}")
        print("-"*80)


@click.command()
@click.option('--start-port', default=10001, help='Starting port number')
@click.option('--end-port', default=10100, help='Ending port number')
@click.option('--num-tests', default=5, help='Number of agents to test randomly')
@click.option('--specific-port', default=0, help='Test a specific agent port')
def main(start_port: int, end_port: int, num_tests: int, specific_port: int):
    """Test multiple agents running on different ports."""
    
    if specific_port > 0:
        # Test specific agent
        try:
            with open('agent-prompts.json', 'r') as f:
                agents = json.load(f)
            
            agent_index = specific_port - 10001
            if 0 <= agent_index < len(agents):
                agent_name = agents[agent_index]['name']
                asyncio.run(test_agent(
                    specific_port, 
                    agent_name,
                    "Tell me about your role and how you can help."
                ))
            else:
                logger.error(f"No agent configured for port {specific_port}")
        except Exception as e:
            logger.error(f"Error testing specific agent: {e}")
    else:
        # Test multiple agents
        asyncio.run(test_multiple_agents(start_port, end_port, num_tests))


if __name__ == '__main__':
    main()
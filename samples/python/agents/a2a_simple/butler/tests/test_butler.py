#!/usr/bin/env python
"""Test the enhanced butler agent"""

import asyncio
import logging
from a2a.client import A2AClient, A2ACardResolver
from a2a.types import MessageSendParams, SendMessageRequest
import httpx
from uuid import uuid4

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_butler():
    """Test the butler agent with a simple query"""
    butler_url = "http://localhost:10000"
    
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as httpx_client:
        try:
            # Get agent card
            resolver = A2ACardResolver(
                httpx_client=httpx_client,
                base_url=butler_url,
            )
            
            logger.info("Getting butler agent card...")
            agent_card = await resolver.get_agent_card()
            logger.info(f"Butler agent: {agent_card.name}")
            logger.info(f"Description: {agent_card.description}")
            logger.info(f"Skills: {[skill.name for skill in agent_card.skills]}")
            
            # Initialize client
            client = A2AClient(
                httpx_client=httpx_client, 
                agent_card=agent_card,
                url=butler_url
            )
            
            # Send a test message
            test_query = "What agents are available and what can they do?"
            logger.info(f"\nSending query: {test_query}")
            
            request = SendMessageRequest(
                id=str(uuid4()),
                params=MessageSendParams(
                    message={
                        'role': 'user',
                        'parts': [{'kind': 'text', 'text': test_query}],
                        'messageId': uuid4().hex,
                    }
                )
            )
            
            response = await client.send_message(request)
            
            # Process response
            if hasattr(response, 'root') and hasattr(response.root, 'result'):
                result = response.root.result
                
                if hasattr(result, 'message') and hasattr(result.message, 'parts'):
                    for part in result.message.parts:
                        if hasattr(part, 'root') and hasattr(part.root, 'text'):
                            logger.info(f"\nButler response:\n{part.root.text}")
                elif hasattr(result, 'state'):
                    logger.info(f"Task state: {result.state}")
                else:
                    logger.info(f"Result: {result}")
            else:
                logger.info(f"Raw response: {response}")
                
        except Exception as e:
            logger.error(f"Error testing butler: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(test_butler())
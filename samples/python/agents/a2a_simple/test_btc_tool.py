#!/usr/bin/env python
"""Test script for BTC price tool integration."""

import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the agent
from app.agent import PromptBasedAgent


async def test_btc_price():
    """Test the BTC price tool functionality."""
    print("Testing BTC price tool integration...\n")
    
    # Create agent with BTC-aware prompt
    btc_prompt = """You are a helpful AI assistant with access to real-time Bitcoin price information.
When users ask about Bitcoin or BTC price, use the get_btc_price tool to fetch current data.
Provide the price information in a clear and concise format."""
    
    agent = PromptBasedAgent(system_prompt=btc_prompt)
    
    # Test queries
    test_queries = [
        "What's the current price of Bitcoin?",
        "Tell me about BTC price and its 24h change",
        "How much is 1 BTC worth in USD?"
    ]
    
    for query in test_queries:
        print(f"Query: {query}")
        print("Response:")
        
        # Stream the response
        async for chunk in agent.stream(query, "test-context"):
            print(f"  {chunk}")
        
        print("\n" + "-" * 50 + "\n")


if __name__ == "__main__":
    asyncio.run(test_btc_price())
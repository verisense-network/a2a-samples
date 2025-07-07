#!/usr/bin/env python3
"""Test script for the butler agent evaluation system"""

import asyncio
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from butler.butler_agent import ButlerAgent
from butler.butler_executor import ButlerAgentExecutor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)


async def test_evaluation_system():
    """Test the evaluation system by running some sample queries"""
    
    # Initialize butler with evaluation
    butler = ButlerAgent()
    
    # Test queries
    test_queries = [
        "What is the current Bitcoin price?",
        "Help me design a new mobile app",
        "Create a Python function to calculate fibonacci numbers",
    ]
    
    print("Testing Butler Agent Evaluation System\n")
    print("=" * 60)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 40)
        
        try:
            # Execute the query
            context_id = "test_" + str(hash(query))
            async for chunk in butler.stream(query, context_id):
                if chunk.get("content"):
                    print(chunk["content"], end="", flush=True)
            
            print("\n")
            
        except Exception as e:
            print(f"Error: {e}")
    
    # Display evaluation statistics
    print("\n" + "=" * 60)
    print("AGENT EVALUATION STATISTICS")
    print("=" * 60)
    
    # Get all agent stats
    all_stats = butler.get_all_agent_stats()
    
    if all_stats:
        print(f"\nTotal agents evaluated: {len(all_stats)}")
        print("\nAgent Performance Rankings:")
        print(f"{'Rank':<5} {'Agent Name':<30} {'Avg Score':<10} {'Total Evals':<12} {'No Response':<12}")
        print("-" * 80)
        
        for i, stats in enumerate(all_stats, 1):
            print(f"{i:<5} {stats['agent_name']:<30} {stats['average_score']:<10.1f} "
                  f"{stats['total_evaluations']:<12} {stats['no_response_count']:<12}")
    else:
        print("No evaluation data available yet.")
    
    # Get recent evaluations
    print("\n" + "=" * 60)
    print("RECENT EVALUATIONS (Last 10)")
    print("=" * 60)
    
    recent_evals = butler.get_recent_evaluations(10)
    
    if recent_evals:
        for eval_record in recent_evals:
            print(f"\nAgent: {eval_record['agent_name']}")
            print(f"Score: {eval_record['score']:.1f}/100")
            print(f"Query: {eval_record['user_query'][:100]}...")
            print(f"Response: {(eval_record['agent_response'] or 'No response')[:100]}...")
            print(f"Reason: {eval_record['evaluation_reason']}")
            print(f"Response Time: {eval_record['response_time_ms']:.0f}ms")
            print(f"Time: {eval_record['timestamp']}")
            print("-" * 40)
    else:
        print("No recent evaluations found.")


async def test_evaluation_api():
    """Test the evaluation API methods"""
    butler = ButlerAgent()
    
    print("\n" + "=" * 60)
    print("TESTING EVALUATION API")
    print("=" * 60)
    
    # Test getting stats for all agents
    all_stats = butler.get_all_agent_stats()
    print(f"\nAll agent stats count: {len(all_stats)}")
    
    # Test getting stats for a specific agent (if any exist)
    if all_stats:
        first_agent_id = all_stats[0]['agent_id']
        agent_stats = butler.get_agent_stats(first_agent_id)
        print(f"\nSpecific agent stats: {json.dumps(agent_stats, indent=2)}")
        
        # Test getting evaluations for this agent
        agent_evals = butler.get_agent_evaluations(first_agent_id, limit=5)
        print(f"\nAgent evaluation history count: {len(agent_evals)}")


if __name__ == "__main__":
    asyncio.run(test_evaluation_system())
    asyncio.run(test_evaluation_api())
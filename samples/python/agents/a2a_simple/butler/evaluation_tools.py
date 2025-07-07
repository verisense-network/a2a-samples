"""Tools for querying agent evaluation statistics"""

from langchain_core.tools import tool
from typing import Dict, List, Any, Optional
from .evaluation import AgentEvaluationDB


# Initialize a shared database instance
_evaluation_db = None

def get_evaluation_db() -> AgentEvaluationDB:
    """Get or create the evaluation database instance"""
    global _evaluation_db
    if _evaluation_db is None:
        _evaluation_db = AgentEvaluationDB()
    return _evaluation_db


@tool
def get_agent_performance_stats(agent_name: Optional[str] = None) -> str:
    """
    Get performance statistics for agents.
    
    Args:
        agent_name: Optional specific agent name to get stats for. If not provided, returns stats for all agents.
    
    Returns:
        Formatted string with agent performance statistics
    """
    db = get_evaluation_db()
    
    if agent_name:
        # Find agent by name
        all_stats = db.get_all_agent_stats()
        agent_stats = None
        for stats in all_stats:
            if stats['agent_name'].lower() == agent_name.lower():
                agent_stats = stats
                break
        
        if not agent_stats:
            return f"No evaluation data found for agent '{agent_name}'"
        
        return f"""
Agent Performance Report: {agent_stats['agent_name']}
=====================================
Average Score: {agent_stats['average_score']:.1f}/100
Total Evaluations: {agent_stats['total_evaluations']}
Best Score: {agent_stats['max_score']:.1f}
Worst Score: {agent_stats['min_score']:.1f}
No Response Count: {agent_stats['no_response_count']}
Response Rate: {((agent_stats['total_evaluations'] - agent_stats['no_response_count']) / agent_stats['total_evaluations'] * 100):.1f}%
Last Updated: {agent_stats['last_updated']}
"""
    else:
        # Get stats for all agents
        all_stats = db.get_all_agent_stats()
        
        if not all_stats:
            return "No evaluation data available yet."
        
        # Format as a leaderboard
        result = "Agent Performance Leaderboard\n"
        result += "=" * 60 + "\n"
        result += f"{'Rank':<5} {'Agent':<25} {'Score':<8} {'Evals':<8} {'Response%':<10}\n"
        result += "-" * 60 + "\n"
        
        for i, stats in enumerate(all_stats[:10], 1):  # Top 10
            response_rate = ((stats['total_evaluations'] - stats['no_response_count']) / stats['total_evaluations'] * 100)
            result += f"{i:<5} {stats['agent_name'][:24]:<25} {stats['average_score']:<8.1f} "
            result += f"{stats['total_evaluations']:<8} {response_rate:<10.1f}\n"
        
        if len(all_stats) > 10:
            result += f"\n... and {len(all_stats) - 10} more agents"
        
        return result


@tool
def get_recent_agent_evaluations(limit: int = 10) -> str:
    """
    Get recent agent evaluation records.
    
    Args:
        limit: Number of recent evaluations to retrieve (default: 10)
    
    Returns:
        Formatted string with recent evaluation details
    """
    db = get_evaluation_db()
    recent_evals = db.get_recent_evaluations(limit)
    
    if not recent_evals:
        return "No recent evaluations found."
    
    result = f"Recent Agent Evaluations (Last {limit})\n"
    result += "=" * 60 + "\n\n"
    
    for eval_record in recent_evals:
        result += f"Agent: {eval_record['agent_name']}\n"
        result += f"Score: {eval_record['score']:.1f}/100\n"
        result += f"Query: {eval_record['user_query'][:80]}...\n"
        
        if eval_record['agent_response']:
            result += f"Response Preview: {eval_record['agent_response'][:80]}...\n"
        else:
            result += "Response: No response received\n"
        
        result += f"Evaluation: {eval_record['evaluation_reason']}\n"
        
        if eval_record['response_time_ms']:
            result += f"Response Time: {eval_record['response_time_ms']:.0f}ms\n"
        
        result += f"Time: {eval_record['timestamp']}\n"
        result += "-" * 40 + "\n\n"
    
    return result


@tool
def get_agent_evaluation_summary() -> str:
    """
    Get a summary of the overall agent evaluation system status.
    
    Returns:
        Summary statistics about the evaluation system
    """
    db = get_evaluation_db()
    all_stats = db.get_all_agent_stats()
    recent_evals = db.get_recent_evaluations(100)
    
    if not all_stats:
        return "Agent evaluation system is active but no evaluations have been recorded yet."
    
    # Calculate overall statistics
    total_agents = len(all_stats)
    total_evaluations = sum(s['total_evaluations'] for s in all_stats)
    total_no_responses = sum(s['no_response_count'] for s in all_stats)
    overall_avg_score = sum(s['average_score'] * s['total_evaluations'] for s in all_stats) / total_evaluations
    
    # Find best and worst performing agents
    best_agent = max(all_stats, key=lambda x: x['average_score'])
    worst_agent = min(all_stats, key=lambda x: x['average_score'])
    
    # Calculate response times if available
    response_times = [e['response_time_ms'] for e in recent_evals if e.get('response_time_ms')]
    avg_response_time = sum(response_times) / len(response_times) if response_times else 0
    
    result = f"""Agent Evaluation System Summary
==============================

Overall Statistics:
- Total Agents Evaluated: {total_agents}
- Total Evaluations: {total_evaluations}
- Overall Average Score: {overall_avg_score:.1f}/100
- Overall Response Rate: {((total_evaluations - total_no_responses) / total_evaluations * 100):.1f}%
- Average Response Time: {avg_response_time:.0f}ms

Top Performer:
- {best_agent['agent_name']}: {best_agent['average_score']:.1f}/100 ({best_agent['total_evaluations']} evals)

Lowest Performer:
- {worst_agent['agent_name']}: {worst_agent['average_score']:.1f}/100 ({worst_agent['total_evaluations']} evals)

Recent Activity:
- Evaluations in last batch: {len(recent_evals)}
"""
    
    return result
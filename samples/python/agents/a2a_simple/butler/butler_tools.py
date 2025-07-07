"""Butler Agent Tools for enhanced orchestration capabilities"""

import json
import logging
from typing import List, Dict, Any, Optional
from langchain_core.tools import tool
import httpx

logger = logging.getLogger(__name__)


@tool
async def query_agent_capabilities(agent_key: str, rpc_url: str = "https://rpc.beta.verisense.network") -> Dict[str, Any]:
    """
    Query detailed capabilities of a specific agent.
    
    Args:
        agent_key: The unique key of the agent to query
        rpc_url: The RPC endpoint URL
        
    Returns:
        Dict containing agent capabilities and metadata
    """
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                rpc_url,
                json={
                    "id": 1,
                    "jsonrpc": "2.0",
                    "method": "a2a_get_agent",
                    "params": [agent_key]
                },
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                data = response.json()
                if "result" in data:
                    return data["result"]
            
            return {"error": f"Failed to query agent {agent_key}"}
            
        except Exception as e:
            logger.error(f"Error querying agent capabilities: {e}")
            return {"error": str(e)}


@tool
async def search_agents_by_skill(skill_tag: str, rpc_url: str = "https://rpc.beta.verisense.network") -> List[Dict[str, Any]]:
    """
    Search for agents that have a specific skill or capability.
    
    Args:
        skill_tag: The skill tag to search for (e.g., "data-analysis", "code-review")
        rpc_url: The RPC endpoint URL
        
    Returns:
        List of agents matching the skill criteria
    """
    async with httpx.AsyncClient() as client:
        try:
            # First get all agents
            response = await client.post(
                rpc_url,
                json={
                    "id": 1,
                    "jsonrpc": "2.0",
                    "method": "a2a_list",
                    "params": []
                },
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                data = response.json()
                if "result" in data:
                    matching_agents = []
                    
                    for item in data["result"]:
                        if isinstance(item, list) and len(item) == 2:
                            key, agent_data = item
                            if isinstance(agent_data, dict):
                                # Check if agent has the skill in description or name
                                name = agent_data.get("name", "").lower()
                                description = agent_data.get("description", "").lower()
                                
                                if skill_tag.lower() in name or skill_tag.lower() in description:
                                    matching_agents.append({
                                        "key": key,
                                        "name": agent_data.get("name", ""),
                                        "url": agent_data.get("url", ""),
                                        "description": agent_data.get("description", "")
                                    })
                    
                    return matching_agents
            
            return []
            
        except Exception as e:
            logger.error(f"Error searching agents by skill: {e}")
            return []


@tool
def create_execution_plan_template(task_description: str, available_agents: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Create a template execution plan for a complex task.
    
    Args:
        task_description: Description of the task to accomplish
        available_agents: List of available agents with their capabilities
        
    Returns:
        A template execution plan that can be customized
    """
    return {
        "task": task_description,
        "plan_template": {
            "phases": [
                {
                    "name": "Analysis Phase",
                    "description": "Understand requirements and constraints",
                    "suggested_agents": [a for a in available_agents if "analyst" in a.get("name", "").lower()],
                    "outputs": ["requirements", "constraints", "success_criteria"]
                },
                {
                    "name": "Design Phase",
                    "description": "Create solution design and architecture",
                    "suggested_agents": [a for a in available_agents if any(role in a.get("name", "").lower() for role in ["architect", "designer", "engineer"])],
                    "outputs": ["design_docs", "architecture", "implementation_plan"]
                },
                {
                    "name": "Implementation Phase",
                    "description": "Execute the implementation",
                    "suggested_agents": [a for a in available_agents if any(role in a.get("name", "").lower() for role in ["developer", "engineer", "programmer"])],
                    "outputs": ["code", "configurations", "documentation"]
                },
                {
                    "name": "Review Phase",
                    "description": "Review and validate the solution",
                    "suggested_agents": [a for a in available_agents if any(role in a.get("name", "").lower() for role in ["reviewer", "tester", "qa"])],
                    "outputs": ["review_feedback", "test_results", "improvements"]
                }
            ],
            "coordination_strategy": "sequential_with_context_passing",
            "error_handling": "retry_with_fallback"
        }
    }


@tool
def analyze_agent_response(response: str) -> Dict[str, Any]:
    """
    Analyze an agent's response to extract key information and artifacts.
    
    Args:
        response: The raw response from an agent
        
    Returns:
        Structured analysis of the response
    """
    analysis = {
        "has_code": "```" in response,
        "has_error": any(word in response.lower() for word in ["error", "failed", "exception"]),
        "has_warning": any(word in response.lower() for word in ["warning", "caution", "note"]),
        "response_type": "unknown",
        "artifacts": [],
        "key_points": []
    }
    
    # Determine response type
    if analysis["has_code"]:
        analysis["response_type"] = "code_generation"
    elif analysis["has_error"]:
        analysis["response_type"] = "error_report"
    elif "plan" in response.lower() or "step" in response.lower():
        analysis["response_type"] = "planning"
    elif "analysis" in response.lower() or "review" in response.lower():
        analysis["response_type"] = "analysis"
    else:
        analysis["response_type"] = "information"
    
    # Extract code artifacts
    if analysis["has_code"]:
        import re
        code_blocks = re.findall(r'```(?:\w+)?\n(.*?)```', response, re.DOTALL)
        for i, code in enumerate(code_blocks):
            analysis["artifacts"].append({
                "type": "code",
                "index": i,
                "content": code.strip()
            })
    
    # Extract key points (lines that start with - or *)
    lines = response.split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith(('-', '*', '•')) and len(line) > 3:
            analysis["key_points"].append(line[2:].strip())
    
    return analysis


@tool
def format_agent_context(agent_name: str, task: str, response: str, include_artifacts: bool = True) -> str:
    """
    Format an agent's response as context for the next agent.
    
    Args:
        agent_name: Name of the agent that provided the response
        task: The task that was given to the agent
        response: The agent's response
        include_artifacts: Whether to include code artifacts in the context
        
    Returns:
        Formatted context string
    """
    context_parts = [
        f"## Previous Agent: {agent_name}",
        f"**Task**: {task}",
        f"**Response Summary**:"
    ]
    
    # Analyze the response
    analysis = analyze_agent_response(response)
    
    # Add key points if found
    if analysis["key_points"]:
        context_parts.append("\n**Key Points**:")
        for point in analysis["key_points"]:
            context_parts.append(f"- {point}")
    
    # Add artifacts if requested and found
    if include_artifacts and analysis["artifacts"]:
        context_parts.append("\n**Generated Artifacts**:")
        for artifact in analysis["artifacts"]:
            if artifact["type"] == "code":
                context_parts.append(f"\n```\n{artifact['content']}\n```")
    
    # Add warnings or errors if found
    if analysis["has_error"]:
        context_parts.append("\n**Note**: The previous agent reported errors or issues.")
    elif analysis["has_warning"]:
        context_parts.append("\n**Note**: The previous agent included warnings.")
    
    return "\n".join(context_parts)


@tool
def create_progress_report(completed_steps: List[Dict[str, str]], remaining_steps: List[Dict[str, str]], overall_status: str) -> str:
    """
    Create a progress report for the current orchestration.
    
    Args:
        completed_steps: List of completed steps with agent and result
        remaining_steps: List of remaining steps
        overall_status: Current status of the overall task
        
    Returns:
        Formatted progress report
    """
    report = [
        "# Orchestration Progress Report",
        f"\n**Overall Status**: {overall_status}",
        f"\n## Completed Steps ({len(completed_steps)})"
    ]
    
    for i, step in enumerate(completed_steps, 1):
        report.append(f"\n### Step {i}: {step.get('task', 'Unknown')}")
        report.append(f"- **Agent**: {step.get('agent', 'Unknown')}")
        report.append(f"- **Status**: ✅ Completed")
        if 'summary' in step:
            report.append(f"- **Summary**: {step['summary']}")
    
    if remaining_steps:
        report.append(f"\n## Remaining Steps ({len(remaining_steps)})")
        for i, step in enumerate(remaining_steps, len(completed_steps) + 1):
            report.append(f"\n### Step {i}: {step.get('task', 'Unknown')}")
            report.append(f"- **Agent**: {step.get('agent', 'TBD')}")
            report.append(f"- **Status**: ⏳ Pending")
    
    report.append("\n---")
    report.append(f"*Progress: {len(completed_steps)}/{len(completed_steps) + len(remaining_steps)} steps completed*")
    
    return "\n".join(report)
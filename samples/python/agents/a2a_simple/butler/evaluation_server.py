#!/usr/bin/env python3
"""Simple HTTP server to view agent evaluation statistics"""

import json
from starlette.applications import Starlette
from starlette.responses import JSONResponse, HTMLResponse
from starlette.routing import Route
import uvicorn
from .evaluation import AgentEvaluationDB
import logging

logger = logging.getLogger(__name__)

# Initialize database
db = AgentEvaluationDB()


async def get_stats(request):
    """API endpoint to get all agent statistics"""
    stats = db.get_all_agent_stats()
    return JSONResponse({
        "status": "success",
        "data": stats,
        "count": len(stats)
    })


async def get_agent_stats(request):
    """API endpoint to get specific agent statistics"""
    agent_id = request.path_params.get("agent_id")
    stats = db.get_agent_stats(agent_id)
    
    if stats:
        return JSONResponse({
            "status": "success",
            "data": stats
        })
    else:
        return JSONResponse({
            "status": "error",
            "message": f"No stats found for agent {agent_id}"
        }, status_code=404)


async def get_recent_evaluations(request):
    """API endpoint to get recent evaluations"""
    limit = int(request.query_params.get("limit", 100))
    evaluations = db.get_recent_evaluations(limit)
    
    # Parse score_breakdown for each evaluation
    for eval_rec in evaluations:
        if eval_rec.get('score_breakdown') and isinstance(eval_rec['score_breakdown'], str):
            try:
                eval_rec['score_breakdown'] = json.loads(eval_rec['score_breakdown'])
            except:
                eval_rec['score_breakdown'] = None
    
    return JSONResponse({
        "status": "success",
        "data": evaluations,
        "count": len(evaluations)
    })


async def get_dashboard(request):
    """Simple HTML dashboard to view evaluation stats"""
    stats = db.get_all_agent_stats()
    recent_evals = db.get_recent_evaluations(20)
    
    # Calculate summary statistics
    total_agents = len(stats)
    total_evaluations = sum(s['total_evaluations'] for s in stats) if stats else 0
    overall_avg = sum(s['average_score'] * s['total_evaluations'] for s in stats) / total_evaluations if total_evaluations > 0 else 0
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Agent Evaluation Dashboard</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
            }}
            h1, h2 {{
                color: #333;
            }}
            .summary {{
                background: white;
                padding: 20px;
                border-radius: 8px;
                margin-bottom: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .stat-card {{
                display: inline-block;
                background: #f0f0f0;
                padding: 15px;
                margin: 10px;
                border-radius: 5px;
                min-width: 150px;
            }}
            table {{
                width: 100%;
                background: white;
                border-collapse: collapse;
                margin-top: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            th, td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #4CAF50;
                color: white;
            }}
            tr:hover {{
                background-color: #f5f5f5;
            }}
            .score {{
                font-weight: bold;
            }}
            .good {{ color: #4CAF50; }}
            .medium {{ color: #FF9800; }}
            .poor {{ color: #F44336; }}
            .no-data {{
                text-align: center;
                padding: 40px;
                color: #666;
            }}
        </style>
        <meta http-equiv="refresh" content="30">
    </head>
    <body>
        <div class="container">
            <h1>🤖 Agent Evaluation Dashboard</h1>
            
            <div class="summary">
                <h2>Summary Statistics</h2>
                <div class="stat-card">
                    <h3>{total_agents}</h3>
                    <p>Total Agents</p>
                </div>
                <div class="stat-card">
                    <h3>{total_evaluations}</h3>
                    <p>Total Evaluations</p>
                </div>
                <div class="stat-card">
                    <h3>{overall_avg:.1f}</h3>
                    <p>Overall Average Score</p>
                </div>
            </div>
            
            <h2>Agent Performance Rankings</h2>
    """
    
    if stats:
        html += """
            <table>
                <tr>
                    <th>Rank</th>
                    <th>Agent Name</th>
                    <th>Average Score</th>
                    <th>Total Evaluations</th>
                    <th>Response Rate</th>
                    <th>Min Score</th>
                    <th>Max Score</th>
                </tr>
        """
        
        for i, stat in enumerate(stats, 1):
            response_rate = ((stat['total_evaluations'] - stat['no_response_count']) / stat['total_evaluations'] * 100) if stat['total_evaluations'] > 0 else 0
            
            # Color code scores
            score_class = 'good' if stat['average_score'] >= 70 else 'medium' if stat['average_score'] >= 40 else 'poor'
            
            html += f"""
                <tr>
                    <td>{i}</td>
                    <td>{stat['agent_name']}</td>
                    <td class="score {score_class}">{stat['average_score']:.1f}</td>
                    <td>{stat['total_evaluations']}</td>
                    <td>{response_rate:.1f}%</td>
                    <td>{stat['min_score']:.1f}</td>
                    <td>{stat['max_score']:.1f}</td>
                </tr>
            """
        
        html += "</table>"
    else:
        html += '<div class="no-data">No evaluation data available yet.</div>'
    
    html += """
            <h2>Recent Evaluations</h2>
    """
    
    if recent_evals:
        html += """
            <table>
                <tr>
                    <th>Time</th>
                    <th>Agent</th>
                    <th>Total Score</th>
                    <th>Score Breakdown</th>
                    <th>Query Preview</th>
                    <th>Response Time</th>
                </tr>
        """
        
        for eval_rec in recent_evals:
            score_class = 'good' if eval_rec['score'] >= 70 else 'medium' if eval_rec['score'] >= 40 else 'poor'
            query_preview = eval_rec['user_query'][:80] + '...' if len(eval_rec['user_query']) > 80 else eval_rec['user_query']
            response_time = f"{eval_rec['response_time_ms']:.0f}ms" if eval_rec['response_time_ms'] else "N/A"
            
            # Parse score breakdown
            breakdown_html = ""
            if eval_rec.get('score_breakdown'):
                try:
                    breakdown = json.loads(eval_rec['score_breakdown']) if isinstance(eval_rec['score_breakdown'], str) else eval_rec['score_breakdown']
                    breakdown_html = f"""
                        R:{breakdown.get('relevance', 0):.0f}/30 
                        C:{breakdown.get('completeness', 0):.0f}/25 
                        A:{breakdown.get('accuracy', 0):.0f}/25 
                        Cl:{breakdown.get('clarity', 0):.0f}/20
                    """
                except:
                    breakdown_html = "N/A"
            else:
                breakdown_html = "N/A"
            
            html += f"""
                <tr>
                    <td>{eval_rec['timestamp']}</td>
                    <td>{eval_rec['agent_name']}</td>
                    <td class="score {score_class}">{eval_rec['score']:.1f}</td>
                    <td style="font-size: 0.9em;">{breakdown_html}</td>
                    <td>{query_preview}</td>
                    <td>{response_time}</td>
                </tr>
            """
        
        html += "</table>"
    else:
        html += '<div class="no-data">No recent evaluations found.</div>'
    
    html += """
            <p style="text-align: center; color: #666; margin-top: 40px;">
                Dashboard auto-refreshes every 30 seconds
            </p>
        </div>
    </body>
    </html>
    """
    
    return HTMLResponse(html)


# Create the application
app = Starlette(debug=True, routes=[
    Route('/', get_dashboard),
    Route('/api/stats', get_stats),
    Route('/api/stats/{agent_id}', get_agent_stats),
    Route('/api/evaluations', get_recent_evaluations),
])


if __name__ == "__main__":
    print("Starting Agent Evaluation Dashboard Server")
    print("Open http://localhost:8888 in your browser to view the dashboard")
    uvicorn.run(app, host="0.0.0.0", port=8888)
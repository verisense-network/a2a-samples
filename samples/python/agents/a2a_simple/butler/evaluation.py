"""Agent Evaluation System for Butler

This module provides functionality to evaluate agent responses and maintain scores
in an embedded SQLite database.
"""

import sqlite3
import json
import datetime
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import threading
from dataclasses import dataclass, asdict
from contextlib import contextmanager

@dataclass
class EvaluationRecord:
    """Represents an evaluation record for an agent response"""
    agent_id: str
    agent_name: str
    task_id: str
    user_query: str
    agent_response: Optional[str]
    score: float  # This is the final weighted score
    evaluation_reason: str
    score_breakdown: Optional[Dict[str, float]]  # 添加评分细节
    current_score: Optional[float] = None  # The raw score for this evaluation
    response_time_ms: Optional[float] = None
    timestamp: datetime.datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['timestamp'] = self.timestamp.isoformat()
        return d


class AgentEvaluationDB:
    """Manages agent evaluation scores in SQLite database"""
    
    def __init__(self, db_path: str = "butler_evaluations.db"):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema"""
        with self._get_connection() as conn:
            # Create table if not exists
            conn.execute("""
                CREATE TABLE IF NOT EXISTS evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_id TEXT NOT NULL,
                    agent_name TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    user_query TEXT NOT NULL,
                    agent_response TEXT,
                    score REAL NOT NULL CHECK (score >= 0 AND score <= 100),
                    evaluation_reason TEXT NOT NULL,
                    score_breakdown TEXT,
                    current_score REAL,
                    response_time_ms REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Check if new columns exist, add them if not
            cursor = conn.execute("PRAGMA table_info(evaluations)")
            columns = [row[1] for row in cursor.fetchall()]
            
            if 'score_breakdown' not in columns:
                try:
                    conn.execute("ALTER TABLE evaluations ADD COLUMN score_breakdown TEXT")
                    conn.commit()
                    print("Added score_breakdown column to evaluations table")
                except Exception as e:
                    # Column might already exist in some edge cases
                    pass
                    
            if 'current_score' not in columns:
                try:
                    conn.execute("ALTER TABLE evaluations ADD COLUMN current_score REAL")
                    conn.commit()
                    print("Added current_score column to evaluations table")
                except Exception as e:
                    # Column might already exist in some edge cases
                    pass
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_agent_id ON evaluations(agent_id)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON evaluations(timestamp)
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS agent_stats (
                    agent_id TEXT PRIMARY KEY,
                    agent_name TEXT NOT NULL,
                    total_evaluations INTEGER DEFAULT 0,
                    average_score REAL DEFAULT 0,
                    min_score REAL DEFAULT 100,
                    max_score REAL DEFAULT 0,
                    no_response_count INTEGER DEFAULT 0,
                    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
    
    @contextmanager
    def _get_connection(self):
        """Get a database connection with thread safety"""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
            finally:
                conn.close()
    
    def add_evaluation(self, record: EvaluationRecord) -> int:
        """Add a new evaluation record"""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                INSERT INTO evaluations (
                    agent_id, agent_name, task_id, user_query, 
                    agent_response, score, evaluation_reason, 
                    score_breakdown, current_score, response_time_ms, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.agent_id, record.agent_name, record.task_id,
                record.user_query, record.agent_response, record.score,
                record.evaluation_reason, 
                json.dumps(record.score_breakdown) if record.score_breakdown else None,
                record.current_score,
                record.response_time_ms,
                record.timestamp
            ))
            
            # Update agent stats
            self._update_agent_stats(conn, record)
            
            conn.commit()
            return cursor.lastrowid
    
    def _update_agent_stats(self, conn, record: EvaluationRecord):
        """Update aggregate statistics for an agent"""
        # Check if agent exists in stats
        existing = conn.execute(
            "SELECT agent_id FROM agent_stats WHERE agent_id = ?",
            (record.agent_id,)
        ).fetchone()
        
        if existing:
            # Update existing stats
            conn.execute("""
                UPDATE agent_stats
                SET total_evaluations = total_evaluations + 1,
                    average_score = (
                        SELECT AVG(score) FROM evaluations 
                        WHERE agent_id = ?
                    ),
                    min_score = MIN(min_score, ?),
                    max_score = MAX(max_score, ?),
                    no_response_count = no_response_count + ?,
                    last_updated = CURRENT_TIMESTAMP
                WHERE agent_id = ?
            """, (
                record.agent_id, record.score, record.score,
                1 if record.agent_response is None else 0,
                record.agent_id
            ))
        else:
            # Insert new stats
            conn.execute("""
                INSERT INTO agent_stats (
                    agent_id, agent_name, total_evaluations,
                    average_score, min_score, max_score,
                    no_response_count
                ) VALUES (?, ?, 1, ?, ?, ?, ?)
            """, (
                record.agent_id, record.agent_name, record.score,
                record.score, record.score,
                1 if record.agent_response is None else 0
            ))
    
    def get_agent_stats(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a specific agent"""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM agent_stats WHERE agent_id = ?",
                (agent_id,)
            ).fetchone()
            
            if row:
                return dict(row)
            return None
    
    def get_all_agent_stats(self) -> List[Dict[str, Any]]:
        """Get statistics for all agents"""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM agent_stats ORDER BY average_score DESC"
            ).fetchall()
            
            return [dict(row) for row in rows]
    
    def get_recent_evaluations(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent evaluation records"""
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT * FROM evaluations 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (limit,)).fetchall()
            
            return [dict(row) for row in rows]
    
    def get_agent_evaluations(self, agent_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get evaluation history for a specific agent"""
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT * FROM evaluations 
                WHERE agent_id = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (agent_id, limit)).fetchall()
            
            return [dict(row) for row in rows]


class AgentEvaluator:
    """Evaluates agent responses using AI"""
    
    def __init__(self, llm_client):
        """Initialize with LLM client (LangChain model)"""
        self.llm = llm_client
    
    async def evaluate_response(
        self, 
        user_query: str, 
        agent_response: Optional[str],
        agent_name: str,
        response_time_ms: Optional[float] = None
    ) -> Tuple[float, str, Dict[str, float]]:
        """
        Evaluate an agent's response quality
        
        Returns:
            Tuple of (score: 0-100, reason: explanation for the score, breakdown: score details)
        """
        if agent_response is None or agent_response.strip() == "":
            breakdown = {
                "relevance": 0,
                "completeness": 0,
                "accuracy": 0,
                "clarity": 0
            }
            return 0.0, "No response received from agent", breakdown
        
        # Create evaluation prompt
        evaluation_prompt = f"""
You are evaluating an AI agent's response quality. Score the response from 0 to 100.

Agent Name: {agent_name}
User Query: {user_query}
Agent Response: {agent_response}

Evaluation Criteria:
1. Relevance (0-30 points): How well does the response address the user's query?
2. Completeness (0-25 points): Is the response thorough and complete?
3. Accuracy (0-25 points): Is the information correct and reliable?
4. Clarity (0-20 points): Is the response clear and well-structured?

Consider response time as a minor factor:
- Response time: {response_time_ms:.0f}ms if provided

Provide your evaluation in the following JSON format:
{{
    "score": <total score 0-100>,
    "breakdown": {{
        "relevance": <0-30>,
        "completeness": <0-25>,
        "accuracy": <0-25>,
        "clarity": <0-20>
    }},
    "reason": "<brief explanation of the score>"
}}
"""
        
        try:
            # Get evaluation from LLM
            response = await self.llm.ainvoke(evaluation_prompt)
            
            # Parse the response
            import re
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                evaluation = json.loads(json_match.group())
                score = float(evaluation.get('score', 50))
                reason = evaluation.get('reason', 'Evaluation completed')
                
                # Ensure score is in valid range
                score = max(0, min(100, score))
                breakdown = evaluation.get('breakdown', {
                    "relevance": 15,
                    "completeness": 12.5,
                    "accuracy": 12.5,
                    "clarity": 10
                })
                
                return score, reason, breakdown
            else:
                # Fallback if parsing fails
                return 50.0, "Could not parse evaluation response", {
                    "relevance": 15,
                    "completeness": 12.5,
                    "accuracy": 12.5,
                    "clarity": 10
                }
                
        except Exception as e:
            # Error in evaluation process
            return 25.0, f"Error during evaluation: {str(e)}", {
                "relevance": 7.5,
                "completeness": 6.25,
                "accuracy": 6.25,
                "clarity": 5
            }
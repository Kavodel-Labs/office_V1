"""
Agora Orchestrator - Core component for multi-LLM consensus
"""

import asyncio
import logging
import json
import uuid
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class AgoraSession:
    """Represents a single Agora consensus session"""
    session_id: str
    task_id: str
    task_type: str
    prompt: str
    context: Dict[str, Any]
    specialists: Dict[str, List[str]]
    max_loops: int = 1
    current_loop: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    total_cost: float = 0.0
    status: str = "initializing"
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if self.start_time:
            data['start_time'] = self.start_time.isoformat()
        if self.end_time:
            data['end_time'] = self.end_time.isoformat()
        return data

@dataclass
class SpecialistResponse:
    """Response from a single LLM specialist"""
    specialist_id: str
    content: str
    reasoning: str
    confidence: float
    tokens_used: int
    latency_ms: int
    cost: float
    role: str = "support"

class AgoraOrchestrator:
    """Orchestrates multi-LLM consensus sessions"""
    
    def __init__(self, config: Dict[str, Any], memory_manager: Any = None):
        self.config = config
        self.memory = memory_manager
        self.specialists = self._load_specialists()
        
        # Import components
        from .merge_engine import MergeEngine
        from .critic_engine import CriticEngine
        from .cost_controller import CostController
        
        self.merge_engine = MergeEngine()
        self.critic_engine = CriticEngine()
        self.cost_controller = CostController(config.get("budget_limits", {}))
        
        # Import LLM engine
        from ..reasoning.multi_llm_interpreter import create_reasoning_engine
        self.llm_engine = create_reasoning_engine(config.get("llm_config", {}))
        
    def _load_specialists(self) -> Dict[str, Dict[str, Any]]:
        """Load specialist configurations"""
        return {
            "synthesizer": {
                "name": "GPT-4o Synthesizer",
                "provider": "openai",
                "model": "gpt-4o",
                "strengths": ["summarization", "clarity_improvement", "documentation"],
                "cost_per_1k_tokens": 0.015,
                "llm_name": "gpt4"
            },
            "builder_logic": {
                "name": "Claude Sonnet Builder",
                "provider": "anthropic", 
                "model": "claude-3-sonnet-20240229",
                "strengths": ["code_generation", "logic_implementation", "debugging"],
                "cost_per_1k_tokens": 0.018,
                "llm_name": "claude"
            },
            "architect": {
                "name": "Gemini Architect",
                "provider": "google",
                "model": "gemini-2.0-flash",
                "strengths": ["system_design", "planning", "scaffolding"],
                "cost_per_1k_tokens": 0.0125,
                "llm_name": "gemini"
            },
            "researcher": {
                "name": "Perplexity Researcher",
                "provider": "perplexity",
                "model": "llama-3.1-sonar-small-128k-online",
                "strengths": ["fact_checking", "research", "citations"],
                "cost_per_1k_tokens": 0.007,
                "llm_name": "perplexity"
            }
        }
    
    async def invoke(self, task: Dict[str, Any], invoking_agent: str) -> Dict[str, Any]:
        """
        Main entry point for Agora consensus
        
        Args:
            task: Task definition including type, prompt, and context
            invoking_agent: ID of the agent requesting consensus
            
        Returns:
            Consensus result with reasoning trace
        """
        # Create session
        session = AgoraSession(
            session_id=str(uuid.uuid4()),
            task_id=task.get("id", str(uuid.uuid4())),
            task_type=task.get("type", "general"),
            prompt=task.get("prompt", ""),
            context=await self._gather_context(task),
            specialists=self._assign_specialists(task),
            max_loops=task.get("max_loops", 1),
            start_time=datetime.utcnow()
        )
        
        logger.info(f"🚀 Starting Agora session {session.session_id} for {task.get('type', 'general')}")
        
        try:
            # Check budget
            if not await self.cost_controller.check_budget(session):
                return self._budget_exceeded_response(session)
            
            # Execute consensus loops
            result = await self._execute_consensus_loop(session)
            
            # Finalize session
            session.end_time = datetime.utcnow()
            session.status = "completed"
            
            # Store results if memory available
            if self.memory:
                await self._store_results(session, result)
            
            logger.info(f"✅ Agora session {session.session_id} completed with confidence {result.get('confidence', 0):.2f}")
            
            return result
            
        except Exception as e:
            session.status = "failed"
            logger.error(f"❌ Agora session {session.session_id} failed: {e}")
            await self._handle_failure(session, e)
            raise
    
    def _assign_specialists(self, task: Dict[str, Any]) -> Dict[str, List[str]]:
        """Assign specialists based on task type"""
        from .role_assignment import RoleAssignment
        
        task_type = task.get("type", "general")
        complexity = task.get("complexity", "medium")
        
        return RoleAssignment.get_specialists(task_type, complexity)
    
    async def _gather_context(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Gather context for the task"""
        context = task.get("context", {})
        
        # Add system context
        context.update({
            "timestamp": datetime.utcnow().isoformat(),
            "available_specialists": list(self.specialists.keys()),
            "task_metadata": {
                "priority": task.get("priority", 5),
                "deadline": task.get("deadline"),
                "budget_limit": task.get("budget_limit")
            }
        })
        
        return context
    
    async def _execute_consensus_loop(self, session: AgoraSession) -> Dict[str, Any]:
        """Execute the main consensus loop"""
        best_result = None
        
        while session.current_loop < session.max_loops:
            session.current_loop += 1
            
            logger.info(f"🔄 Agora loop {session.current_loop}/{session.max_loops} for {session.session_id}")
            
            # Phase 1: Parallel specialist execution
            specialist_responses = await self._execute_specialists(session)
            
            if not specialist_responses:
                logger.warning(f"No valid specialist responses in session {session.session_id}")
                break
            
            # Phase 2: Merge responses
            merged_result = await self.merge_engine.merge(specialist_responses, session)
            
            # Phase 3: Critic evaluation
            critique = await self.critic_engine.evaluate(merged_result, session)
            
            # Phase 4: Decide if refinement needed
            quality_threshold = self.config.get("quality_threshold", 0.8)
            if critique["quality_score"] >= quality_threshold:
                best_result = {
                    "content": merged_result["content"],
                    "reasoning_trace": {
                        "session_id": session.session_id,
                        "loops": session.current_loop,
                        "specialist_responses": [r.specialist_id for r in specialist_responses],
                        "merged_result": merged_result,
                        "critique": critique
                    },
                    "confidence": critique["quality_score"],
                    "total_cost": session.total_cost,
                    "method": "agora_consensus"
                }
                break
            
            # Update context for next loop
            session.context["previous_attempt"] = merged_result
            session.context["critique"] = critique
        
        return best_result or self._fallback_result(session)
    
    async def _execute_specialists(self, session: AgoraSession) -> List[SpecialistResponse]:
        """Execute all specialists in parallel"""
        tasks = []
        
        # Create tasks for each specialist
        for role, specialist_ids in session.specialists.items():
            if role == "critic":  # Skip critic in this phase
                continue
                
            for specialist_id in specialist_ids:
                if specialist_id in self.specialists:
                    specialist = self.specialists[specialist_id]
                    task = self._execute_single_specialist(specialist, session, role)
                    tasks.append(task)
        
        if not tasks:
            logger.warning(f"No valid specialist tasks for session {session.session_id}")
            return []
        
        # Execute all tasks in parallel
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out errors and return valid responses
        valid_responses = []
        for response in responses:
            if isinstance(response, SpecialistResponse):
                valid_responses.append(response)
                session.total_cost += response.cost
            else:
                # Log error but continue
                logger.error(f"Specialist error in session {session.session_id}: {response}")
        
        return valid_responses
    
    async def _execute_single_specialist(self, specialist: Dict[str, Any], 
                                       session: AgoraSession, role: str) -> SpecialistResponse:
        """Execute a single LLM specialist"""
        start_time = time.time()
        
        try:
            # Build specialist-specific prompt
            prompt = self._build_specialist_prompt(specialist, session, role)
            
            # Prepare context for LLM engine
            llm_context = {
                "user_id": session.session_id,
                "channel": "agora",
                "specialist_role": role,
                "task_type": session.task_type
            }
            
            # Call LLM through our engine
            if hasattr(self.llm_engine, 'interpreters'):
                # Find the appropriate interpreter
                interpreter = None
                for interp in self.llm_engine.interpreters:
                    if hasattr(interp, 'name') and specialist["llm_name"] in interp.name.lower():
                        interpreter = interp
                        break
                
                if interpreter:
                    interpretation = await interpreter.interpret(prompt, llm_context)
                    content = interpretation.response or interpretation.intent
                    reasoning = interpretation.reasoning or ""
                    confidence = interpretation.confidence
                else:
                    # Fallback
                    content = f"Specialist {specialist['name']} response for: {session.prompt}"
                    reasoning = f"Processed with {specialist['name']} capabilities"
                    confidence = 0.7
            else:
                # Simple fallback
                content = f"Specialist {specialist['name']} response for: {session.prompt}"
                reasoning = f"Processed with {specialist['name']} capabilities"
                confidence = 0.7
            
            # Calculate metrics
            latency_ms = int((time.time() - start_time) * 1000)
            tokens_used = len(prompt.split()) + len(content.split())  # Rough estimate
            cost = (tokens_used / 1000) * specialist["cost_per_1k_tokens"]
            
            return SpecialistResponse(
                specialist_id=specialist["name"],
                content=content,
                reasoning=reasoning,
                confidence=confidence,
                tokens_used=tokens_used,
                latency_ms=latency_ms,
                cost=cost,
                role=role
            )
            
        except Exception as e:
            logger.error(f"Error executing specialist {specialist['name']}: {e}")
            # Return error response
            latency_ms = int((time.time() - start_time) * 1000)
            return SpecialistResponse(
                specialist_id=specialist["name"],
                content=f"Error: {str(e)}",
                reasoning="Specialist execution failed",
                confidence=0.0,
                tokens_used=0,
                latency_ms=latency_ms,
                cost=0.0,
                role=role
            )
    
    def _build_specialist_prompt(self, specialist: Dict[str, Any], 
                               session: AgoraSession, role: str) -> str:
        """Build role-specific prompt for specialist"""
        base_prompt = f"""You are {specialist['name']}, a specialized AI with expertise in {', '.join(specialist['strengths'])}.

Your role in this task is: {role}

Task Type: {session.task_type}
Task Prompt: {session.prompt}

Context:
{json.dumps(session.context, indent=2)}

Please provide your response following these guidelines:
1. Focus on your areas of expertise: {', '.join(specialist['strengths'])}
2. Be specific and detailed
3. Provide clear reasoning for your suggestions
4. Consider the role you're playing ({role})

Response should be practical and actionable."""
        
        # Add role-specific instructions
        if role == "lead":
            base_prompt += "\n\nAs the lead specialist, provide a comprehensive solution that integrates your expertise."
        elif role == "support":
            base_prompt += "\n\nAs a supporting specialist, focus on your specific expertise area and how it enhances the solution."
            
        # Add loop-specific context
        if session.current_loop > 1:
            base_prompt += f"\n\nPrevious Attempt:\n{session.context.get('previous_attempt', {}).get('content', 'N/A')}"
            base_prompt += f"\n\nCritique:\n{session.context.get('critique', {}).get('feedback', 'N/A')}"
            
        return base_prompt
    
    def _budget_exceeded_response(self, session: AgoraSession) -> Dict[str, Any]:
        """Return response when budget is exceeded"""
        return {
            "content": "Budget limit exceeded for this consensus request",
            "confidence": 0.0,
            "total_cost": 0.0,
            "method": "budget_limited",
            "reasoning_trace": {
                "session_id": session.session_id,
                "status": "budget_exceeded"
            }
        }
    
    def _fallback_result(self, session: AgoraSession) -> Dict[str, Any]:
        """Return fallback result when consensus fails"""
        return {
            "content": f"Unable to reach consensus for task: {session.prompt}",
            "confidence": 0.3,
            "total_cost": session.total_cost,
            "method": "fallback",
            "reasoning_trace": {
                "session_id": session.session_id,
                "loops": session.current_loop,
                "status": "consensus_failed"
            }
        }
    
    async def _store_results(self, session: AgoraSession, result: Dict[str, Any]):
        """Store session results in memory"""
        if self.memory:
            try:
                # Store in hot memory
                await self.memory.hot.set(
                    f"agora:session:{session.session_id}",
                    {
                        "status": session.status,
                        "result": result["content"],
                        "confidence": result["confidence"],
                        "cost": session.total_cost
                    },
                    ttl=3600
                )
                logger.info(f"Stored Agora session {session.session_id} in memory")
            except Exception as e:
                logger.error(f"Failed to store session results: {e}")
    
    async def _handle_failure(self, session: AgoraSession, error: Exception):
        """Handle session failure"""
        logger.error(f"Agora session {session.session_id} failed: {error}")
        
        if self.memory:
            try:
                await self.memory.hot.set(
                    f"agora:session:{session.session_id}",
                    {
                        "status": "failed",
                        "error": str(error),
                        "cost": session.total_cost
                    },
                    ttl=1800
                )
            except Exception as e:
                logger.error(f"Failed to store failure info: {e}")
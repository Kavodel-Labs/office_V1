"""
Cost Controller for Agora Consensus
Manages budget limits and cost optimization for LLM consensus sessions
"""

import logging
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class BudgetMetrics:
    """Budget tracking metrics"""
    total_budget: float
    used_budget: float
    remaining_budget: float
    session_count: int
    avg_cost_per_session: float
    daily_limit: float
    daily_used: float

class CostController:
    """Manages budget limits and cost optimization"""
    
    def __init__(self, budget_config: Dict[str, Any]):
        self.budget_config = budget_config
        self.session_costs = {}  # Track costs per session
        self.daily_costs = {}   # Track daily spending
        
        # Default budget limits
        self.limits = {
            "total_daily": budget_config.get("daily_limit", 10.0),
            "per_session": budget_config.get("session_limit", 2.0),
            "per_loop": budget_config.get("loop_limit", 0.5),
            "emergency_threshold": budget_config.get("emergency_threshold", 0.1)
        }
        
        # Cost optimization settings
        self.optimization = {
            "auto_reduce_complexity": budget_config.get("auto_reduce_complexity", True),
            "prefer_cheaper_models": budget_config.get("prefer_cheaper_models", False),
            "max_loops_budget_aware": budget_config.get("max_loops_budget_aware", True)
        }
        
    async def check_budget(self, session) -> bool:
        """
        Check if session can proceed within budget constraints
        
        Returns:
            True if session can proceed, False if budget exceeded
        """
        try:
            # Get current budget metrics
            metrics = await self.get_budget_metrics()
            
            # Check daily limit
            if metrics.daily_used >= metrics.daily_limit:
                logger.warning(f"Daily budget limit exceeded: ${metrics.daily_used:.3f} >= ${metrics.daily_limit:.3f}")
                return False
            
            # Estimate session cost
            estimated_cost = await self._estimate_session_cost(session)
            
            # Check if session would exceed limits
            if estimated_cost > self.limits["per_session"]:
                logger.warning(f"Session estimated cost exceeds limit: ${estimated_cost:.3f} > ${self.limits['per_session']:.3f}")
                return False
            
            # Check if session would exceed daily budget
            if metrics.daily_used + estimated_cost > metrics.daily_limit:
                logger.warning(f"Session would exceed daily budget: ${metrics.daily_used + estimated_cost:.3f} > ${metrics.daily_limit:.3f}")
                return False
            
            # Emergency threshold check
            remaining_ratio = metrics.remaining_budget / metrics.total_budget
            if remaining_ratio < self.limits["emergency_threshold"]:
                logger.warning(f"Emergency budget threshold reached: {remaining_ratio:.3f} < {self.limits['emergency_threshold']:.3f}")
                return False
            
            logger.info(f"Budget check passed for session {session.session_id}: estimated ${estimated_cost:.3f}")
            return True
            
        except Exception as e:
            logger.error(f"Budget check failed: {e}")
            # Fail safe - allow if we can't check
            return True
    
    async def optimize_session(self, session) -> Dict[str, Any]:
        """
        Optimize session configuration for cost efficiency
        
        Returns:
            Optimized session parameters
        """
        try:
            optimization_params = {}
            metrics = await self.get_budget_metrics()
            
            # Adjust complexity based on remaining budget
            if self.optimization["auto_reduce_complexity"]:
                budget_ratio = metrics.remaining_budget / metrics.total_budget
                
                if budget_ratio < 0.2:  # Less than 20% budget remaining
                    optimization_params["complexity"] = "low"
                    optimization_params["max_loops"] = 1
                    logger.info(f"Reduced complexity to 'low' due to budget constraints")
                elif budget_ratio < 0.5:  # Less than 50% budget remaining
                    optimization_params["complexity"] = "medium"
                    optimization_params["max_loops"] = min(session.max_loops, 2)
                    logger.info(f"Reduced complexity to 'medium' due to budget constraints")
            
            # Prefer cheaper models if enabled
            if self.optimization["prefer_cheaper_models"]:
                optimization_params["preferred_specialists"] = self._get_budget_optimized_specialists()
            
            # Adjust loop limits based on budget
            if self.optimization["max_loops_budget_aware"]:
                remaining_for_session = min(
                    self.limits["per_session"],
                    metrics.daily_limit - metrics.daily_used
                )
                max_affordable_loops = int(remaining_for_session / self.limits["per_loop"])
                optimization_params["max_loops"] = min(session.max_loops, max_affordable_loops)
                logger.info(f"Budget-aware max loops: {optimization_params.get('max_loops', session.max_loops)}")
            
            return optimization_params
            
        except Exception as e:
            logger.error(f"Session optimization failed: {e}")
            return {}
    
    async def track_session_cost(self, session_id: str, cost: float):
        """Track actual cost for a session"""
        self.session_costs[session_id] = cost
        
        # Track daily costs
        today = datetime.now().strftime("%Y-%m-%d")
        if today not in self.daily_costs:
            self.daily_costs[today] = 0.0
        self.daily_costs[today] += cost
        
        logger.info(f"Tracked session {session_id} cost: ${cost:.3f}")
    
    async def get_budget_metrics(self) -> BudgetMetrics:
        """Get current budget metrics"""
        # Calculate totals
        total_used = sum(self.session_costs.values())
        session_count = len(self.session_costs)
        avg_cost = total_used / max(session_count, 1)
        
        # Daily costs
        today = datetime.now().strftime("%Y-%m-%d")
        daily_used = self.daily_costs.get(today, 0.0)
        daily_limit = self.limits["total_daily"]
        
        # Total budget (estimated based on daily limit)
        total_budget = daily_limit * 30  # Monthly estimate
        remaining_budget = total_budget - total_used
        
        return BudgetMetrics(
            total_budget=total_budget,
            used_budget=total_used,
            remaining_budget=max(0, remaining_budget),
            session_count=session_count,
            avg_cost_per_session=avg_cost,
            daily_limit=daily_limit,
            daily_used=daily_used
        )
    
    async def get_cost_report(self) -> Dict[str, Any]:
        """Generate detailed cost report"""
        metrics = await self.get_budget_metrics()
        
        # Calculate trends
        recent_sessions = list(self.session_costs.items())[-10:]  # Last 10 sessions
        recent_avg = sum(cost for _, cost in recent_sessions) / max(len(recent_sessions), 1)
        
        # Daily breakdown for last 7 days
        daily_breakdown = {}
        for i in range(7):
            date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
            daily_breakdown[date] = self.daily_costs.get(date, 0.0)
        
        return {
            "budget_metrics": {
                "total_budget": metrics.total_budget,
                "used_budget": metrics.used_budget,
                "remaining_budget": metrics.remaining_budget,
                "budget_utilization": metrics.used_budget / metrics.total_budget,
                "daily_utilization": metrics.daily_used / metrics.daily_limit
            },
            "session_metrics": {
                "total_sessions": metrics.session_count,
                "avg_cost_per_session": metrics.avg_cost_per_session,
                "recent_avg_cost": recent_avg,
                "most_expensive_session": max(self.session_costs.values()) if self.session_costs else 0.0,
                "least_expensive_session": min(self.session_costs.values()) if self.session_costs else 0.0
            },
            "daily_breakdown": daily_breakdown,
            "optimization_settings": self.optimization,
            "budget_limits": self.limits
        }
    
    async def _estimate_session_cost(self, session) -> float:
        """Estimate cost for a session based on configuration"""
        try:
            # Base cost estimation per specialist
            specialist_costs = {
                "synthesizer": 0.25,      # GPT-4o
                "builder_logic": 0.30,    # Claude Sonnet
                "architect": 0.20,        # Gemini
                "researcher": 0.15        # Perplexity
            }
            
            total_estimated = 0.0
            
            # Calculate cost based on assigned specialists
            for role, specialist_ids in session.specialists.items():
                for specialist_id in specialist_ids:
                    base_cost = specialist_costs.get(specialist_id, 0.25)
                    # Multiply by expected loops
                    total_estimated += base_cost * session.max_loops
            
            # Add overhead for merge and critic operations
            overhead = total_estimated * 0.1
            total_estimated += overhead
            
            logger.debug(f"Estimated session cost for {session.session_id}: ${total_estimated:.3f}")
            return total_estimated
            
        except Exception as e:
            logger.error(f"Cost estimation failed: {e}")
            return 0.5  # Conservative fallback estimate
    
    def _get_budget_optimized_specialists(self) -> List[str]:
        """Get list of specialists ordered by cost efficiency"""
        # Order by cost (cheapest first)
        return ["researcher", "architect", "synthesizer", "builder_logic"]
    
    async def set_budget_limits(self, new_limits: Dict[str, float]):
        """Update budget limits"""
        self.limits.update(new_limits)
        logger.info(f"Updated budget limits: {self.limits}")
    
    async def reset_daily_budget(self):
        """Reset daily budget tracking (called at start of new day)"""
        today = datetime.now().strftime("%Y-%m-%d")
        if today not in self.daily_costs:
            self.daily_costs[today] = 0.0
        logger.info(f"Daily budget reset for {today}")
    
    async def emergency_budget_stop(self) -> bool:
        """Check if emergency budget stop should be triggered"""
        metrics = await self.get_budget_metrics()
        emergency_triggered = (
            metrics.remaining_budget <= 0 or
            metrics.daily_used >= metrics.daily_limit * 1.1 or  # 110% of daily limit
            metrics.used_budget / metrics.total_budget > 0.95   # 95% of total budget
        )
        
        if emergency_triggered:
            logger.critical(f"EMERGENCY BUDGET STOP TRIGGERED - Budget metrics: {metrics}")
        
        return emergency_triggered
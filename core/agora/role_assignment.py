"""
Role Assignment for Agora Specialists
Maps task types to optimal LLM specialist teams
"""

from typing import Dict, List

class RoleAssignment:
    """Maps task types to optimal LLM specialist teams"""
    
    TASK_SPECIALIST_MAP = {
        "ui_component": {
            "lead": "builder_logic",
            "support": ["architect", "synthesizer"],
            "critic": "synthesizer"
        },
        "api_endpoint": {
            "lead": "builder_logic",
            "support": ["architect", "researcher"],
            "critic": "synthesizer"
        },
        "system_design": {
            "lead": "architect",
            "support": ["builder_logic", "researcher"],
            "critic": "synthesizer"
        },
        "documentation": {
            "lead": "synthesizer",
            "support": ["researcher"],
            "critic": "builder_logic"
        },
        "bug_investigation": {
            "lead": "builder_logic",
            "support": ["researcher", "architect"],
            "critic": "synthesizer"
        },
        "code_generation": {
            "lead": "builder_logic",
            "support": ["architect", "synthesizer"],
            "critic": "researcher"
        },
        "research_request": {
            "lead": "researcher",
            "support": ["synthesizer", "architect"],
            "critic": "builder_logic"
        },
        "planning": {
            "lead": "architect",
            "support": ["synthesizer", "researcher"],
            "critic": "builder_logic"
        },
        "analysis": {
            "lead": "researcher",
            "support": ["builder_logic", "synthesizer"],
            "critic": "architect"
        },
        "creative": {
            "lead": "synthesizer",
            "support": ["architect", "researcher"],
            "critic": "builder_logic"
        },
        "general": {
            "lead": "synthesizer",
            "support": ["architect"],
            "critic": "builder_logic"
        },
        "natural_language_analysis": {
            "lead": "synthesizer",
            "support": ["builder_logic", "researcher"],
            "critic": "architect"
        },
        "slack_interpretation": {
            "lead": "synthesizer",
            "support": ["researcher"],
            "critic": "builder_logic"
        }
    }
    
    @classmethod
    def get_specialists(cls, task_type: str, complexity: str = "medium") -> Dict[str, List[str]]:
        """Return optimal specialist configuration for task"""
        base_config = cls.TASK_SPECIALIST_MAP.get(task_type, cls.TASK_SPECIALIST_MAP["general"])
        
        # Create a copy to avoid modifying the original
        config = {
            "lead": [base_config["lead"]],
            "support": base_config["support"].copy(),
            "critic": [base_config["critic"]]
        }
        
        if complexity == "high":
            # Add more specialists for complex tasks
            if "researcher" not in config["support"]:
                config["support"].append("researcher")
        elif complexity == "low":
            # Reduce specialists for simple tasks
            config["support"] = config["support"][:1]
            
        return config
    
    @classmethod
    def get_task_types(cls) -> List[str]:
        """Get all supported task types"""
        return list(cls.TASK_SPECIALIST_MAP.keys())
    
    @classmethod
    def get_specialist_strengths(cls, specialist_id: str) -> List[str]:
        """Get strengths for a specialist"""
        strengths_map = {
            "synthesizer": ["summarization", "clarity_improvement", "documentation", "communication"],
            "builder_logic": ["code_generation", "logic_implementation", "debugging", "problem_solving"],
            "architect": ["system_design", "planning", "scaffolding", "architecture"],
            "researcher": ["fact_checking", "research", "citations", "data_analysis"]
        }
        return strengths_map.get(specialist_id, [])
    
    @classmethod
    def recommend_task_type(cls, description: str) -> str:
        """Recommend task type based on description"""
        description_lower = description.lower()
        
        # Keywords mapping
        if any(word in description_lower for word in ["ui", "interface", "component", "frontend"]):
            return "ui_component"
        elif any(word in description_lower for word in ["api", "endpoint", "backend", "service"]):
            return "api_endpoint"
        elif any(word in description_lower for word in ["design", "architecture", "system", "structure"]):
            return "system_design"
        elif any(word in description_lower for word in ["document", "write", "explain", "manual"]):
            return "documentation"
        elif any(word in description_lower for word in ["bug", "error", "fix", "debug", "issue"]):
            return "bug_investigation"
        elif any(word in description_lower for word in ["code", "implement", "function", "class"]):
            return "code_generation"
        elif any(word in description_lower for word in ["research", "find", "investigate", "analyze"]):
            return "research_request"
        elif any(word in description_lower for word in ["plan", "strategy", "roadmap", "timeline"]):
            return "planning"
        elif any(word in description_lower for word in ["creative", "innovative", "brainstorm", "idea"]):
            return "creative"
        elif any(word in description_lower for word in ["slack", "message", "interpret", "analyze text"]):
            return "natural_language_analysis"
        else:
            return "general"
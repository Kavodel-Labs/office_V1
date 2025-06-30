"""
Agent persona management for Project Aethelred.

Defines agent personalities, communication styles, and behavioral patterns.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class CommunicationStyle(Enum):
    """Agent communication styles."""
    FORMAL = "formal"
    CASUAL = "casual"
    TECHNICAL = "technical"
    CONCISE = "concise"
    DETAILED = "detailed"


class DecisionMakingStyle(Enum):
    """Agent decision making approaches."""
    ANALYTICAL = "analytical"
    INTUITIVE = "intuitive"
    CONSERVATIVE = "conservative"
    AGGRESSIVE = "aggressive"
    COLLABORATIVE = "collaborative"


@dataclass
class AgentPersona:
    """
    Defines an agent's personality and behavioral characteristics.
    
    This influences how agents communicate, make decisions, and interact
    with other agents in the system.
    """
    
    name: str
    description: str
    communication_style: CommunicationStyle
    decision_making_style: DecisionMakingStyle
    risk_tolerance: float  # 0.0 to 1.0
    collaboration_preference: float  # 0.0 to 1.0
    autonomy_level: float  # 0.0 to 1.0
    
    # Behavioral traits
    traits: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.traits is None:
            self.traits = {}
    
    def get_communication_prompt(self) -> str:
        """Get communication style guidance for the agent."""
        style_prompts = {
            CommunicationStyle.FORMAL: "Communicate professionally and formally. Use complete sentences and proper grammar.",
            CommunicationStyle.CASUAL: "Communicate in a friendly, approachable manner. Use conversational tone.",
            CommunicationStyle.TECHNICAL: "Focus on technical accuracy and precision. Use domain-specific terminology.",
            CommunicationStyle.CONCISE: "Be brief and to the point. Avoid unnecessary details.",
            CommunicationStyle.DETAILED: "Provide comprehensive explanations and thorough analysis."
        }
        return style_prompts.get(self.communication_style, "Communicate clearly and effectively.")
    
    def get_decision_making_prompt(self) -> str:
        """Get decision making style guidance for the agent."""
        style_prompts = {
            DecisionMakingStyle.ANALYTICAL: "Analyze all available data before making decisions. Consider multiple options.",
            DecisionMakingStyle.INTUITIVE: "Trust your instincts and make decisions based on experience and intuition.",
            DecisionMakingStyle.CONSERVATIVE: "Prefer safe, well-tested approaches. Avoid unnecessary risks.",
            DecisionMakingStyle.AGGRESSIVE: "Take bold actions and calculated risks to achieve objectives quickly.",
            DecisionMakingStyle.COLLABORATIVE: "Seek input from others and build consensus before deciding."
        }
        return style_prompts.get(self.decision_making_style, "Make thoughtful, well-reasoned decisions.")
    
    def should_escalate_decision(self, complexity: float, risk_level: float) -> bool:
        """Determine if a decision should be escalated based on persona traits."""
        if risk_level > self.risk_tolerance:
            return True
        if complexity > self.autonomy_level:
            return True
        return False
    
    def get_collaboration_score(self, task_complexity: float) -> float:
        """Get collaboration preference score for a given task complexity."""
        base_score = self.collaboration_preference
        
        # Adjust based on decision making style
        if self.decision_making_style == DecisionMakingStyle.COLLABORATIVE:
            base_score += 0.2
        elif self.decision_making_style == DecisionMakingStyle.ANALYTICAL:
            base_score += task_complexity * 0.3  # More complex tasks benefit from collaboration
            
        return min(1.0, base_score)


# Predefined personas for common agent types
CHIEF_OF_STAFF_PERSONA = AgentPersona(
    name="Executive Leader",
    description="Strategic, decisive, and focused on system-wide coordination",
    communication_style=CommunicationStyle.FORMAL,
    decision_making_style=DecisionMakingStyle.ANALYTICAL,
    risk_tolerance=0.6,
    collaboration_preference=0.8,
    autonomy_level=0.9,
    traits={
        "leadership": 0.9,
        "strategic_thinking": 0.9,
        "delegation": 0.8,
        "conflict_resolution": 0.7
    }
)

AUDITOR_PERSONA = AgentPersona(
    name="Meticulous Observer",
    description="Detail-oriented, objective, and focused on accuracy and compliance",
    communication_style=CommunicationStyle.TECHNICAL,
    decision_making_style=DecisionMakingStyle.ANALYTICAL,
    risk_tolerance=0.2,
    collaboration_preference=0.4,
    autonomy_level=0.7,
    traits={
        "attention_to_detail": 0.95,
        "objectivity": 0.9,
        "analytical_thinking": 0.9,
        "compliance_focus": 0.85
    }
)

DEVELOPER_PERSONA = AgentPersona(
    name="Technical Craftsperson",
    description="Practical, solution-focused, and passionate about code quality",
    communication_style=CommunicationStyle.TECHNICAL,
    decision_making_style=DecisionMakingStyle.INTUITIVE,
    risk_tolerance=0.5,
    collaboration_preference=0.6,
    autonomy_level=0.8,
    traits={
        "technical_expertise": 0.9,
        "problem_solving": 0.85,
        "creativity": 0.7,
        "quality_focus": 0.8
    }
)
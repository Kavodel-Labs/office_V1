"""
Critic Engine for Agora Consensus
Evaluates merged results and provides improvement feedback
"""

import logging
import json
import re
from typing import Dict, Any

logger = logging.getLogger(__name__)

class CriticEngine:
    """Evaluates merged results and provides improvement feedback"""
    
    def __init__(self):
        self.evaluation_criteria = {
            "completeness": self._evaluate_completeness,
            "correctness": self._evaluate_correctness_heuristic,
            "clarity": self._evaluate_clarity_heuristic,
            "efficiency": self._evaluate_efficiency_heuristic,
            "creativity": self._evaluate_creativity_heuristic
        }
        
    async def evaluate(self, merged_result: Dict[str, Any], session) -> Dict[str, Any]:
        """
        Evaluate merged result against task requirements
        
        Returns:
            Critique with quality score and specific feedback
        """
        try:
            # Get critic specialist
            critic_id = session.specialists.get("critic", ["synthesizer"])[0]
            
            # Build evaluation prompt
            eval_prompt = self._build_evaluation_prompt(merged_result, session)
            
            # Get evaluation using rule-based approach + LLM if available
            evaluation = await self._perform_evaluation(eval_prompt, merged_result, session)
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(evaluation, session.task_type)
            
            result = {
                "quality_score": quality_score,
                "feedback": evaluation.get("feedback", "Evaluation completed"),
                "strengths": evaluation.get("strengths", []),
                "weaknesses": evaluation.get("weaknesses", []),
                "improvement_suggestions": evaluation.get("suggestions", []),
                "meets_requirements": quality_score >= 0.8,
                "evaluation_method": evaluation.get("method", "rule_based")
            }
            
            logger.info(f"Critic evaluation completed with quality score: {quality_score:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Critic evaluation failed: {e}")
            # Return conservative evaluation
            return {
                "quality_score": 0.7,
                "feedback": f"Evaluation error: {str(e)}",
                "strengths": ["Response provided"],
                "weaknesses": ["Could not fully evaluate"],
                "improvement_suggestions": ["Review and verify manually"],
                "meets_requirements": False,
                "evaluation_method": "error_fallback"
            }
    
    def _build_evaluation_prompt(self, merged_result: Dict[str, Any], session) -> str:
        """Build prompt for critic evaluation"""
        return f"""
You are acting as an impartial critic evaluating a solution.

Original Task: {session.prompt}
Task Type: {session.task_type}

Proposed Solution:
{merged_result['content']}

Consensus Areas:
{json.dumps(merged_result.get('consensus', []), indent=2)}

Conflict Areas:
{json.dumps(merged_result.get('conflicts', []), indent=2)}

Specialist Contributions:
{json.dumps(merged_result.get('specialist_contributions', []), indent=2)}

Please evaluate this solution across these dimensions:
1. Completeness - Does it fully address the task requirements?
2. Correctness - Is the solution technically sound?
3. Clarity - Is it well-organized and easy to understand?
4. Efficiency - Is it optimal in terms of performance/resources?
5. Creativity - Does it show innovative thinking?

Provide:
- Specific strengths
- Specific weaknesses
- Concrete improvement suggestions
- Overall assessment

Format your response as JSON with these keys:
strengths, weaknesses, suggestions, feedback, scores (object with dimension scores 0-1)
"""
    
    async def _perform_evaluation(self, eval_prompt: str, merged_result: Dict[str, Any], session) -> Dict[str, Any]:
        """Perform the actual evaluation"""
        # Rule-based evaluation as primary method
        rule_based_eval = self._rule_based_evaluation(merged_result, session)
        
        # Try to enhance with LLM evaluation if available
        try:
            llm_eval = await self._llm_based_evaluation(eval_prompt, session)
            if llm_eval:
                # Combine rule-based and LLM evaluations
                return self._combine_evaluations(rule_based_eval, llm_eval)
        except Exception as e:
            logger.warning(f"LLM evaluation failed, using rule-based: {e}")
        
        return rule_based_eval
    
    def _rule_based_evaluation(self, merged_result: Dict[str, Any], session) -> Dict[str, Any]:
        """Rule-based evaluation using heuristics"""
        content = merged_result.get("content", "")
        consensus = merged_result.get("consensus", [])
        conflicts = merged_result.get("conflicts", [])
        contributions = merged_result.get("specialist_contributions", [])
        
        # Calculate scores for each dimension
        scores = {}
        
        # Completeness - based on content length and structure
        scores["completeness"] = self._evaluate_completeness(content, session)
        
        # Correctness - based on consensus vs conflicts
        scores["correctness"] = self._evaluate_correctness_heuristic(consensus, conflicts, contributions)
        
        # Clarity - based on structure and readability
        scores["clarity"] = self._evaluate_clarity_heuristic(content)
        
        # Efficiency - based on content optimization
        scores["efficiency"] = self._evaluate_efficiency_heuristic(content, session.task_type)
        
        # Creativity - based on novel approaches
        scores["creativity"] = self._evaluate_creativity_heuristic(content, contributions)
        
        # Generate feedback
        strengths = self._identify_strengths(scores, merged_result)
        weaknesses = self._identify_weaknesses(scores, merged_result)
        suggestions = self._generate_suggestions(scores, session.task_type)
        
        return {
            "scores": scores,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "suggestions": suggestions,
            "feedback": self._generate_feedback(scores, strengths, weaknesses),
            "method": "rule_based"
        }
    
    async def _llm_based_evaluation(self, eval_prompt: str, session) -> Dict[str, Any]:
        """LLM-based evaluation if available"""
        # This would use the actual LLM critics - simplified for now
        return None
    
    def _combine_evaluations(self, rule_eval: Dict[str, Any], llm_eval: Dict[str, Any]) -> Dict[str, Any]:
        """Combine rule-based and LLM evaluations"""
        # Weight rule-based more heavily for reliability
        combined_scores = {}
        for criterion in rule_eval["scores"]:
            rule_score = rule_eval["scores"][criterion]
            llm_score = llm_eval.get("scores", {}).get(criterion, rule_score)
            combined_scores[criterion] = rule_score * 0.7 + llm_score * 0.3
        
        return {
            "scores": combined_scores,
            "strengths": rule_eval["strengths"] + llm_eval.get("strengths", []),
            "weaknesses": rule_eval["weaknesses"] + llm_eval.get("weaknesses", []),
            "suggestions": rule_eval["suggestions"] + llm_eval.get("suggestions", []),
            "feedback": rule_eval["feedback"],
            "method": "combined"
        }
    
    # Evaluation dimension methods
    def _evaluate_completeness(self, content: str, session) -> float:
        """Evaluate completeness based on content analysis"""
        if not content or len(content.strip()) < 50:
            return 0.2
        
        # Check for structure indicators
        structure_score = 0.0
        if "##" in content or "#" in content:  # Has headers
            structure_score += 0.3
        if len(content.split('\n')) > 5:  # Multi-line content
            structure_score += 0.2
        if any(keyword in content.lower() for keyword in ["solution", "approach", "implementation"]):
            structure_score += 0.3
        
        # Length-based scoring
        length_score = min(len(content) / 500, 1.0) * 0.2
        
        return min(structure_score + length_score, 1.0)
    
    def _evaluate_correctness_heuristic(self, consensus: list, conflicts: list, contributions: list) -> float:
        """Evaluate correctness based on consensus analysis"""
        if not contributions:
            return 0.5
        
        # High consensus with few conflicts indicates correctness
        consensus_ratio = len(consensus) / max(len(consensus) + len(conflicts), 1)
        
        # High confidence from specialists
        avg_confidence = sum(c.get("confidence", 0.5) for c in contributions) / len(contributions)
        
        # Combine factors
        return (consensus_ratio * 0.6 + avg_confidence * 0.4)
    
    def _evaluate_clarity_heuristic(self, content: str) -> float:
        """Evaluate clarity based on structure and readability"""
        if not content:
            return 0.0
        
        clarity_score = 0.0
        
        # Structure indicators
        if re.search(r'^#{1,3}\s', content, re.MULTILINE):  # Headers
            clarity_score += 0.3
        if '```' in content:  # Code blocks
            clarity_score += 0.2
        if re.search(r'^\s*[•\-\*]\s', content, re.MULTILINE):  # Bullet points
            clarity_score += 0.2
        
        # Readability (simple metrics)
        sentences = content.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        if 10 <= avg_sentence_length <= 25:  # Reasonable sentence length
            clarity_score += 0.3
        
        return min(clarity_score, 1.0)
    
    def _evaluate_efficiency_heuristic(self, content: str, task_type: str) -> float:
        """Evaluate efficiency based on task type and content"""
        # Task-specific efficiency indicators
        efficiency_keywords = {
            "code": ["optimize", "efficient", "performance", "fast"],
            "design": ["scalable", "maintainable", "modular"],
            "documentation": ["concise", "clear", "structured"],
            "research": ["comprehensive", "thorough", "accurate"]
        }
        
        relevant_keywords = []
        for category, keywords in efficiency_keywords.items():
            if category in task_type.lower():
                relevant_keywords = keywords
                break
        
        if not relevant_keywords:
            relevant_keywords = efficiency_keywords["design"]  # Default
        
        # Count efficiency indicators
        efficiency_mentions = sum(1 for keyword in relevant_keywords if keyword in content.lower())
        efficiency_score = min(efficiency_mentions / len(relevant_keywords), 1.0)
        
        # Penalize overly verbose solutions
        if len(content) > 2000:
            efficiency_score *= 0.8
        
        return efficiency_score
    
    def _evaluate_creativity_heuristic(self, content: str, contributions: list) -> float:
        """Evaluate creativity based on novel approaches"""
        creativity_score = 0.0
        
        # Diversity of specialist input
        if len(contributions) > 2:
            creativity_score += 0.3
        
        # Creative language indicators
        creative_keywords = ["innovative", "novel", "creative", "unique", "alternative", "new approach"]
        creativity_mentions = sum(1 for keyword in creative_keywords if keyword in content.lower())
        creativity_score += min(creativity_mentions * 0.2, 0.4)
        
        # Multiple approaches mentioned
        if "approach" in content.lower() and content.lower().count("approach") > 1:
            creativity_score += 0.3
        
        return min(creativity_score, 1.0)
    
    def _calculate_quality_score(self, evaluation: Dict[str, Any], task_type: str) -> float:
        """Calculate overall quality score with task-specific weighting"""
        scores = evaluation.get("scores", {})
        
        # Default weights
        weights = {
            "completeness": 0.25,
            "correctness": 0.25,
            "clarity": 0.20,
            "efficiency": 0.15,
            "creativity": 0.15
        }
        
        # Adjust weights based on task type
        if "bug" in task_type or "debug" in task_type:
            weights["correctness"] = 0.40
            weights["creativity"] = 0.05
        elif "ui" in task_type or "creative" in task_type:
            weights["creativity"] = 0.30
            weights["efficiency"] = 0.10
        elif "documentation" in task_type or "explain" in task_type:
            weights["clarity"] = 0.35
            weights["creativity"] = 0.10
        elif "research" in task_type or "analyze" in task_type:
            weights["completeness"] = 0.35
            weights["correctness"] = 0.30
        
        # Calculate weighted score
        total_score = sum(
            scores.get(criterion, 0.5) * weight 
            for criterion, weight in weights.items()
        )
        
        return max(0.0, min(1.0, total_score))
    
    def _identify_strengths(self, scores: Dict[str, float], merged_result: Dict[str, Any]) -> list:
        """Identify strengths based on scores"""
        strengths = []
        
        for criterion, score in scores.items():
            if score >= 0.8:
                strengths.append(f"Strong {criterion} - demonstrates {self._get_strength_description(criterion)}")
        
        # Additional strengths from merged result
        consensus = merged_result.get("consensus", [])
        if len(consensus) > 2:
            strengths.append("High specialist consensus on key points")
        
        contributions = merged_result.get("specialist_contributions", [])
        if len(contributions) > 2:
            strengths.append("Diverse specialist perspectives incorporated")
        
        return strengths or ["Solution addresses the core requirements"]
    
    def _identify_weaknesses(self, scores: Dict[str, float], merged_result: Dict[str, Any]) -> list:
        """Identify weaknesses based on scores"""
        weaknesses = []
        
        for criterion, score in scores.items():
            if score < 0.6:
                weaknesses.append(f"Limited {criterion} - could improve {self._get_weakness_description(criterion)}")
        
        # Additional weaknesses from merged result
        conflicts = merged_result.get("conflicts", [])
        if len(conflicts) > 1:
            weaknesses.append("Significant disagreement between specialists on some points")
        
        return weaknesses
    
    def _generate_suggestions(self, scores: Dict[str, float], task_type: str) -> list:
        """Generate improvement suggestions"""
        suggestions = []
        
        for criterion, score in scores.items():
            if score < 0.7:
                suggestions.append(self._get_improvement_suggestion(criterion, task_type))
        
        # General suggestions
        if not any(score > 0.8 for score in scores.values()):
            suggestions.append("Consider requesting additional specialist review")
        
        return suggestions or ["Solution is well-developed"]
    
    def _generate_feedback(self, scores: Dict[str, float], strengths: list, weaknesses: list) -> str:
        """Generate overall feedback"""
        avg_score = sum(scores.values()) / len(scores)
        
        if avg_score >= 0.8:
            tone = "Excellent"
        elif avg_score >= 0.7:
            tone = "Good"
        elif avg_score >= 0.6:
            tone = "Satisfactory"
        else:
            tone = "Needs improvement"
        
        feedback = f"{tone} solution with an overall quality score of {avg_score:.2f}. "
        
        if strengths:
            feedback += f"Key strengths include {strengths[0].lower()}. "
        
        if weaknesses:
            feedback += f"Primary area for improvement: {weaknesses[0].lower()}."
        
        return feedback
    
    # Helper methods for descriptions
    def _get_strength_description(self, criterion: str) -> str:
        descriptions = {
            "completeness": "comprehensive coverage of requirements",
            "correctness": "technical accuracy and reliability",
            "clarity": "clear structure and communication", 
            "efficiency": "optimal approach and resource usage",
            "creativity": "innovative thinking and novel solutions"
        }
        return descriptions.get(criterion, "high quality")
    
    def _get_weakness_description(self, criterion: str) -> str:
        descriptions = {
            "completeness": "more comprehensive coverage",
            "correctness": "technical accuracy verification",
            "clarity": "structure and readability",
            "efficiency": "optimization and resource usage",
            "creativity": "innovative approaches"
        }
        return descriptions.get(criterion, "overall quality")
    
    def _get_improvement_suggestion(self, criterion: str, task_type: str) -> str:
        suggestions = {
            "completeness": "Add more detailed coverage of edge cases and requirements",
            "correctness": "Verify technical accuracy and validate approach",
            "clarity": "Improve structure with headers, examples, and clear explanations",
            "efficiency": "Consider optimization opportunities and resource efficiency",
            "creativity": "Explore alternative approaches and innovative solutions"
        }
        return suggestions.get(criterion, "Review and enhance the solution")
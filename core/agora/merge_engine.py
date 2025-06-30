"""
Merge Engine for Agora Consensus
Synthesizes multiple specialist responses into unified solutions
"""

import logging
import re
import json
from typing import List, Dict, Any
from collections import Counter
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ConsensusAnalysis:
    """Analysis of consensus between specialist responses"""
    consensus_points: List[Dict[str, Any]]
    conflict_points: List[Dict[str, Any]]
    overall_confidence: float
    agreement_ratio: float

class MergeEngine:
    """Merges multiple specialist responses into a unified solution"""
    
    def __init__(self):
        self.merge_strategies = {
            "code": self._merge_code,
            "design": self._merge_design,
            "documentation": self._merge_documentation,
            "research": self._merge_research,
            "general": self._merge_general
        }
        
    async def merge(self, responses: List, session) -> Dict[str, Any]:
        """
        Merge specialist responses based on task type
        
        Returns:
            Merged result with consensus areas and conflicts identified
        """
        if not responses:
            return {
                "content": "No specialist responses to merge",
                "consensus": [],
                "conflicts": [],
                "confidence": 0.0,
                "specialist_contributions": []
            }
        
        # Determine merge strategy
        task_category = self._categorize_task(session.task_type)
        merge_strategy = self.merge_strategies.get(task_category, self._merge_general)
        
        # Execute merge
        merged_content = merge_strategy(responses)
        
        # Analyze consensus and conflicts
        consensus_analysis = self._analyze_consensus(responses)
        
        # Extract contributions
        contributions = self._extract_contributions(responses)
        
        result = {
            "content": merged_content,
            "consensus": consensus_analysis.consensus_points,
            "conflicts": consensus_analysis.conflict_points,
            "confidence": consensus_analysis.overall_confidence,
            "specialist_contributions": contributions,
            "agreement_ratio": consensus_analysis.agreement_ratio
        }
        
        logger.info(f"Merged {len(responses)} specialist responses with {consensus_analysis.overall_confidence:.2f} confidence")
        
        return result
    
    def _categorize_task(self, task_type: str) -> str:
        """Categorize task type for merge strategy selection"""
        if any(keyword in task_type.lower() for keyword in ["code", "implement", "function", "api"]):
            return "code"
        elif any(keyword in task_type.lower() for keyword in ["design", "architecture", "system"]):
            return "design"
        elif any(keyword in task_type.lower() for keyword in ["document", "write", "explain"]):
            return "documentation"
        elif any(keyword in task_type.lower() for keyword in ["research", "analyze", "investigate"]):
            return "research"
        else:
            return "general"
    
    def _merge_code(self, responses: List) -> str:
        """Merge code-based responses"""
        logger.info("Merging code responses")
        
        # Extract code blocks from each response
        all_code_blocks = []
        explanations = []
        
        for response in responses:
            code_blocks = self._extract_code_blocks(response.content)
            all_code_blocks.extend(code_blocks)
            
            # Extract explanations
            explanation = self._extract_explanation(response.content)
            if explanation:
                explanations.append(f"**{response.specialist_id}**: {explanation}")
        
        # Find the most comprehensive code solution
        best_code = self._select_best_code(all_code_blocks, responses)
        
        # Combine with explanations
        merged = f"""## Solution

{best_code}

## Specialist Insights

{chr(10).join(explanations)}

## Implementation Notes

This solution synthesizes approaches from multiple specialists, focusing on correctness, clarity, and best practices."""
        
        return merged
    
    def _merge_design(self, responses: List) -> str:
        """Merge design/architecture responses"""
        logger.info("Merging design responses")
        
        # Extract design components from each response
        all_components = []
        design_rationales = []
        
        for response in responses:
            components = self._extract_components(response.content)
            all_components.extend(components)
            
            rationale = self._extract_rationale(response.content)
            if rationale:
                design_rationales.append(f"**{response.specialist_id}**: {rationale}")
        
        # Merge components by category
        merged_components = self._merge_components(all_components)
        
        merged = f"""## System Design

### Components
{self._format_components(merged_components)}

### Design Rationale
{chr(10).join(design_rationales)}

### Architecture Overview
This design synthesizes multiple expert perspectives to create a robust, scalable solution."""
        
        return merged
    
    def _merge_documentation(self, responses: List) -> str:
        """Merge documentation responses"""
        logger.info("Merging documentation responses")
        
        # Extract sections from each response
        all_sections = {}
        for response in responses:
            sections = self._extract_sections(response.content)
            for section_name, content in sections.items():
                if section_name not in all_sections:
                    all_sections[section_name] = []
                all_sections[section_name].append({
                    "content": content,
                    "specialist": response.specialist_id,
                    "confidence": response.confidence
                })
        
        # Merge sections by taking the best content for each
        merged_sections = []
        for section_name, contents in all_sections.items():
            best_content = max(contents, key=lambda x: x["confidence"])
            merged_sections.append(f"## {section_name}\n\n{best_content['content']}")
        
        # Add contributor acknowledgment
        contributors = list(set(r.specialist_id for r in responses))
        
        merged = f"""{chr(10).join(merged_sections)}

---
*Documentation compiled from insights by: {', '.join(contributors)}*"""
        
        return merged
    
    def _merge_research(self, responses: List) -> str:
        """Merge research responses"""
        logger.info("Merging research responses")
        
        # Combine all findings
        all_findings = []
        sources = []
        
        for response in responses:
            findings = self._extract_findings(response.content)
            all_findings.extend(findings)
            
            response_sources = self._extract_sources(response.content)
            sources.extend(response_sources)
        
        # Remove duplicates and organize
        unique_findings = list(set(all_findings))
        unique_sources = list(set(sources))
        
        merged = f"""## Research Summary

### Key Findings
{chr(10).join(f"• {finding}" for finding in unique_findings)}

### Sources and References
{chr(10).join(f"• {source}" for source in unique_sources)}

### Methodology
This research synthesis combines multiple specialist approaches including fact-checking, analysis, and citation verification."""
        
        return merged
    
    def _merge_general(self, responses: List) -> str:
        """Merge general responses"""
        logger.info("Merging general responses")
        
        # Weight responses by confidence and role
        weighted_responses = []
        for response in responses:
            weight = response.confidence
            if response.role == "lead":
                weight *= 1.5
            weighted_responses.append((response, weight))
        
        # Sort by weight
        weighted_responses.sort(key=lambda x: x[1], reverse=True)
        
        # Take the top response as base and enhance with others
        base_response = weighted_responses[0][0]
        merged = f"""## Solution

{base_response.content}

## Additional Perspectives

"""
        
        # Add insights from other specialists
        for response, weight in weighted_responses[1:]:
            if weight > 0.5:  # Only include high-confidence responses
                merged += f"**{response.specialist_id}**: {self._extract_key_insight(response.content)}\n\n"
        
        return merged
    
    def _analyze_consensus(self, responses: List) -> ConsensusAnalysis:
        """Analyze level of agreement between specialists"""
        if not responses:
            return ConsensusAnalysis([], [], 0.0, 0.0)
        
        # Extract key points from each response
        all_points = []
        for response in responses:
            points = self._extract_key_points(response.content)
            all_points.extend([(point, response.specialist_id, response.confidence) for point in points])
        
        # Find consensus and conflicts
        point_counts = Counter([point[0] for point in all_points])
        total_specialists = len(responses)
        
        consensus_points = []
        conflict_points = []
        
        for point, count in point_counts.items():
            agreement_ratio = count / total_specialists
            supporters = [p[1] for p in all_points if p[0] == point]
            avg_confidence = sum(p[2] for p in all_points if p[0] == point) / count
            
            point_data = {
                "point": point,
                "agreement_ratio": agreement_ratio,
                "supporters": supporters,
                "confidence": avg_confidence
            }
            
            if agreement_ratio >= 0.7:  # 70% agreement threshold
                consensus_points.append(point_data)
            elif agreement_ratio <= 0.3:  # Significant disagreement
                conflict_points.append(point_data)
        
        # Calculate overall metrics
        overall_confidence = sum(r.confidence for r in responses) / len(responses)
        avg_agreement = sum(p["agreement_ratio"] for p in consensus_points) / max(len(consensus_points), 1)
        
        return ConsensusAnalysis(
            consensus_points=consensus_points,
            conflict_points=conflict_points,
            overall_confidence=overall_confidence,
            agreement_ratio=avg_agreement
        )
    
    def _extract_contributions(self, responses: List) -> List[Dict[str, Any]]:
        """Extract individual specialist contributions"""
        contributions = []
        for response in responses:
            contributions.append({
                "specialist": response.specialist_id,
                "role": response.role,
                "confidence": response.confidence,
                "key_insight": self._extract_key_insight(response.content),
                "tokens_used": response.tokens_used,
                "cost": response.cost
            })
        return contributions
    
    # Helper methods for content extraction
    def _extract_code_blocks(self, content: str) -> List[str]:
        """Extract code blocks from content"""
        # Match code blocks with ``` or indented blocks
        code_pattern = r'```[\w]*\n(.*?)\n```|^[ ]{4,}(.+)$'
        matches = re.findall(code_pattern, content, re.MULTILINE | re.DOTALL)
        return [match[0] or match[1] for match in matches if match[0] or match[1]]
    
    def _extract_explanation(self, content: str) -> str:
        """Extract explanation text (non-code content)"""
        # Remove code blocks and get remaining text
        content_without_code = re.sub(r'```[\w]*\n.*?\n```', '', content, flags=re.DOTALL)
        return content_without_code.strip()[:200] + "..." if len(content_without_code) > 200 else content_without_code.strip()
    
    def _select_best_code(self, code_blocks: List[str], responses: List) -> str:
        """Select the best code block based on length and specialist confidence"""
        if not code_blocks:
            return "# No code provided"
        
        # Simple heuristic: longest code block from highest confidence specialist
        best_code = max(code_blocks, key=len) if code_blocks else "# No code available"
        return best_code
    
    def _extract_components(self, content: str) -> List[str]:
        """Extract design components from content"""
        # Look for component mentions (simple pattern matching)
        component_patterns = [
            r'component[s]?:?\s*([^\n]+)',
            r'module[s]?:?\s*([^\n]+)',
            r'service[s]?:?\s*([^\n]+)',
            r'class[es]*:?\s*([^\n]+)'
        ]
        components = []
        for pattern in component_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            components.extend(matches)
        return components
    
    def _extract_rationale(self, content: str) -> str:
        """Extract design rationale"""
        # Look for reasoning/rationale sections
        rationale_patterns = [
            r'rationale:?\s*([^\n]{1,200})',
            r'because\s+([^\n]{1,200})',
            r'reason:?\s*([^\n]{1,200})'
        ]
        for pattern in rationale_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return content[:100] + "..." if len(content) > 100 else content
    
    def _merge_components(self, components: List[str]) -> Dict[str, List[str]]:
        """Merge and categorize components"""
        categorized = {"core": [], "support": [], "other": []}
        for component in components:
            if any(keyword in component.lower() for keyword in ["main", "core", "primary"]):
                categorized["core"].append(component)
            elif any(keyword in component.lower() for keyword in ["util", "helper", "support"]):
                categorized["support"].append(component)
            else:
                categorized["other"].append(component)
        return categorized
    
    def _format_components(self, components: Dict[str, List[str]]) -> str:
        """Format components for display"""
        formatted = []
        for category, items in components.items():
            if items:
                formatted.append(f"**{category.title()}**: {', '.join(items)}")
        return "\n".join(formatted)
    
    def _extract_sections(self, content: str) -> Dict[str, str]:
        """Extract sections from documentation"""
        # Split by headers (## or #)
        sections = {}
        current_section = "Introduction"
        current_content = []
        
        lines = content.split('\n')
        for line in lines:
            if line.startswith('##') or line.startswith('#'):
                # Save previous section
                if current_content:
                    sections[current_section] = '\n'.join(current_content).strip()
                
                # Start new section
                current_section = line.strip('#').strip()
                current_content = []
            else:
                current_content.append(line)
        
        # Save last section
        if current_content:
            sections[current_section] = '\n'.join(current_content).strip()
        
        return sections
    
    def _extract_findings(self, content: str) -> List[str]:
        """Extract research findings"""
        # Look for bullet points or numbered lists
        findings = []
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('•') or line.startswith('-') or re.match(r'^\d+\.', line):
                findings.append(line)
        return findings
    
    def _extract_sources(self, content: str) -> List[str]:
        """Extract sources and references"""
        # Look for URLs, citations, or reference patterns
        url_pattern = r'https?://[^\s]+'
        citation_pattern = r'\[.*?\]|\(.*?\)'
        
        sources = []
        sources.extend(re.findall(url_pattern, content))
        sources.extend(re.findall(citation_pattern, content))
        
        return sources
    
    def _extract_key_points(self, content: str) -> List[str]:
        """Extract key points from content for consensus analysis"""
        # Simple extraction of sentences or bullet points
        points = []
        
        # Split by sentences
        sentences = re.split(r'[.!?]+', content)
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20 and len(sentence) < 150:  # Reasonable length
                points.append(sentence)
        
        # Also get bullet points
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('•') or line.startswith('-'):
                points.append(line)
        
        return points[:5]  # Limit to top 5 points
    
    def _extract_key_insight(self, content: str) -> str:
        """Extract the key insight from a response"""
        # Get the first substantial sentence or paragraph
        sentences = content.split('.')
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 30:
                return sentence + "."
        
        # Fallback to first 150 characters
        return content[:150] + "..." if len(content) > 150 else content
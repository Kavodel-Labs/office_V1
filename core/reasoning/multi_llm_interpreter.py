"""
Multi-LLM Consensus-Based Interpretation Engine for AETHELRED
Implementation of the reasoning layer from aethelred-slack-interface specification.
"""

import asyncio
import logging
import json
import time
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from collections import Counter
from abc import ABC, abstractmethod
import aiohttp
import openai
from anthropic import Anthropic
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None

logger = logging.getLogger(__name__)

@dataclass
class Interpretation:
    """Structured interpretation result from LLM analysis."""
    intent: str
    entities: Dict[str, Any]
    confidence: float
    response: Optional[str] = None
    reasoning: Optional[str] = None
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

@dataclass
class ConsensusResult:
    """Result of multi-LLM consensus analysis."""
    final_interpretation: Interpretation
    individual_results: List[Interpretation]
    consensus_confidence: float
    disagreement_areas: List[str]
    processing_time: float

class BaseLLMInterpreter(ABC):
    """Base class for LLM interpreters."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = config.get("name", "unknown")
        self.model = config.get("model", "unknown")
        self.temperature = config.get("temperature", 0.3)
        self.timeout = config.get("timeout", 10)
        
    @abstractmethod
    async def interpret(self, message: str, context: Dict[str, Any]) -> Interpretation:
        """Interpret a message and return structured result."""
        pass
    
    def _build_interpretation_prompt(self, message: str, context: Dict[str, Any]) -> str:
        """Build standardized interpretation prompt."""
        return f"""You are an AI interpreter for the AETHELRED autonomous AI workforce system.

Context:
- User ID: {context.get('user_id', 'unknown')}
- Channel: {context.get('channel', 'unknown')}
- Previous messages: {context.get('history', [])[-3:]}  # Last 3 for context
- Current time: {time.strftime('%Y-%m-%d %H:%M:%S')}

Message to interpret: "{message}"

Analyze this message and respond with ONLY valid JSON in this exact format:
{{
    "intent": "one of: status_inquiry, task_creation, terminal_command, research_request, configuration_change, clarification_needed, greeting, help_request, decision_response, file_request",
    "entities": {{
        "target": "what the message is about (system, agent, task, etc.)",
        "action": "what should be done",
        "priority": "low|medium|high",
        "parameters": {{}}
    }},
    "confidence": 0.85,
    "suggested_response": "A helpful, professional response to the user",
    "reasoning": "Brief explanation of why you chose this interpretation"
}}

IMPORTANT: 
- Respond with ONLY the JSON object, no other text
- Confidence should be 0.0-1.0 based on how certain you are
- Use "clarification_needed" intent if the message is unclear
- For terminal commands, extract the actual command in parameters.command
- For greetings, use intent "greeting"
"""

class GPT4Interpreter(BaseLLMInterpreter):
    """OpenAI GPT-4 interpreter."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Set API key for openai module (v0.x compatibility)
        openai.api_key = config.get("api_key")
        
    async def interpret(self, message: str, context: Dict[str, Any]) -> Interpretation:
        """Interpret using GPT-4."""
        try:
            prompt = self._build_interpretation_prompt(message, context)
            
            # Use openai v0.x API format
            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=500,
                request_timeout=self.timeout
            )
            
            content = response.choices[0].message.content.strip()
            return self._parse_response(content)
            
        except Exception as e:
            logger.error(f"GPT-4 interpretation failed: {e}")
            return Interpretation(
                intent="error",
                entities={"error": str(e), "source": "gpt4"},
                confidence=0.0,
                response="I'm having trouble understanding right now. Please try again."
            )

class ClaudeInterpreter(BaseLLMInterpreter):
    """Anthropic Claude interpreter."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.client = Anthropic(api_key=config.get("api_key"))
        
    async def interpret(self, message: str, context: Dict[str, Any]) -> Interpretation:
        """Interpret using Claude."""
        try:
            prompt = self._build_interpretation_prompt(message, context)
            
            # Note: Making this sync for now as anthropic client doesn't have async yet
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.messages.create(
                    model=self.model,
                    max_tokens=500,
                    temperature=self.temperature,
                    messages=[{"role": "user", "content": prompt}]
                )
            )
            
            content = response.content[0].text.strip()
            return self._parse_response(content)
            
        except Exception as e:
            logger.error(f"Claude interpretation failed: {e}")
            return Interpretation(
                intent="error",
                entities={"error": str(e), "source": "claude"},
                confidence=0.0,
                response="I'm having trouble understanding right now. Please try again."
            )

class LocalLlamaInterpreter(BaseLLMInterpreter):
    """Local Llama model interpreter via Ollama."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.endpoint = config.get("endpoint", "http://localhost:11434")
        
    async def interpret(self, message: str, context: Dict[str, Any]) -> Interpretation:
        """Interpret using local Llama model."""
        try:
            prompt = self._build_interpretation_prompt(message, context)
            
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": 500
                    }
                }
                
                async with session.post(
                    f"{self.endpoint}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    result = await response.json()
                    content = result.get("response", "").strip()
                    return self._parse_response(content)
                    
        except Exception as e:
            logger.error(f"Local Llama interpretation failed: {e}")
            return Interpretation(
                intent="error",
                entities={"error": str(e), "source": "local_llama"},
                confidence=0.0,
                response="I'm having trouble understanding right now. Please try again."
            )

    def _parse_response(self, response: str) -> Interpretation:
        """Parse LLM response into Interpretation object."""
        try:
            # Extract JSON from response (handle cases where LLM adds extra text)
            response = response.strip()
            if not response.startswith('{'):
                # Find first { and last }
                start = response.find('{')
                end = response.rfind('}') + 1
                if start != -1 and end > start:
                    response = response[start:end]
                else:
                    raise ValueError("No valid JSON found in response")
            
            data = json.loads(response)
            
            return Interpretation(
                intent=data.get("intent", "unknown"),
                entities=data.get("entities", {}),
                confidence=float(data.get("confidence", 0.5)),
                response=data.get("suggested_response"),
                reasoning=data.get("reasoning")
            )
            
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            logger.debug(f"Raw response: {response}")
            
            # Fallback interpretation
            return Interpretation(
                intent="clarification_needed",
                entities={"parse_error": str(e)},
                confidence=0.1,
                response="I need clarification - could you rephrase your request?"
            )

class GeminiInterpreter(BaseLLMInterpreter):
    """Google Gemini interpreter using REST API."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get("api_key")
        # Use the new gemini-2.0-flash model
        self.model = "gemini-2.0-flash"
        self.endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        
    async def interpret(self, message: str, context: Dict[str, Any]) -> Interpretation:
        """Interpret using Gemini REST API."""
        try:
            prompt = self._build_interpretation_prompt(message, context)
            
            payload = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": prompt
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": self.temperature,
                    "maxOutputTokens": 500
                }
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.endpoint}?key={self.api_key}",
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    result = await response.json()
                    
                    if "candidates" in result and result["candidates"]:
                        content = result["candidates"][0]["content"]["parts"][0]["text"].strip()
                        return self._parse_response(content)
                    else:
                        raise Exception(f"No candidates in Gemini response: {result}")
                    
        except Exception as e:
            logger.error(f"Gemini interpretation failed: {e}")
            return Interpretation(
                intent="error",
                entities={"error": str(e), "source": "gemini"},
                confidence=0.0,
                response="I'm having trouble understanding right now. Please try again."
            )

class PerplexityInterpreter(BaseLLMInterpreter):
    """Perplexity AI interpreter."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get("api_key")
        self.endpoint = "https://api.perplexity.ai/chat/completions"
        
    async def interpret(self, message: str, context: Dict[str, Any]) -> Interpretation:
        """Interpret using Perplexity."""
        try:
            prompt = self._build_interpretation_prompt(message, context)
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
                "max_tokens": 500
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.endpoint,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    result = await response.json()
                    content = result["choices"][0]["message"]["content"].strip()
                    return self._parse_response(content)
                    
        except Exception as e:
            logger.error(f"Perplexity interpretation failed: {e}")
            return Interpretation(
                intent="error",
                entities={"error": str(e), "source": "perplexity"},
                confidence=0.0,
                response="I'm having trouble understanding right now. Please try again."
            )

# Add this method to BaseLLMInterpreter
BaseLLMInterpreter._parse_response = lambda self, response: LocalLlamaInterpreter._parse_response(self, response)

class MultiLLMReasoningEngine:
    """
    Multi-LLM consensus-based reasoning engine.
    Implements the core intelligence for AETHELRED Slack interface.
    """
    
    def __init__(self, llm_configs: List[Dict[str, Any]]):
        self.interpreters = []
        self.consensus_threshold = 0.8
        self.clarification_threshold = 0.5
        self.max_interpretation_time = 10  # seconds
        
        # Initialize LLM interpreters
        for config in llm_configs:
            try:
                if config["name"] == "gpt4":
                    self.interpreters.append(GPT4Interpreter(config))
                elif config["name"] == "claude":
                    self.interpreters.append(ClaudeInterpreter(config))
                elif config["name"] == "gemini":
                    self.interpreters.append(GeminiInterpreter(config))
                elif config["name"] == "perplexity":
                    self.interpreters.append(PerplexityInterpreter(config))
                elif config["name"] == "local_llama":
                    self.interpreters.append(LocalLlamaInterpreter(config))
                else:
                    logger.warning(f"Unknown LLM type: {config['name']}")
            except Exception as e:
                logger.error(f"Failed to initialize {config.get('name', 'unknown')} interpreter: {e}")
        
        if not self.interpreters:
            logger.warning("No LLM interpreters initialized - falling back to simple pattern matching")
            
        logger.info(f"Initialized MultiLLM engine with {len(self.interpreters)} interpreters")
    
    async def interpret_with_consensus(self, message: str, context: Dict[str, Any]) -> ConsensusResult:
        """
        Get consensus interpretation from multiple LLMs.
        Core method implementing the reasoning layer specification.
        """
        start_time = time.time()
        
        if not self.interpreters:
            # Fallback to simple pattern matching
            simple_result = self._simple_pattern_matching(message, context)
            return ConsensusResult(
                final_interpretation=simple_result,
                individual_results=[simple_result],
                consensus_confidence=0.3,
                disagreement_areas=["no_llms_available"],
                processing_time=time.time() - start_time
            )
        
        try:
            # Run all interpreters in parallel with timeout
            tasks = [
                asyncio.wait_for(
                    interpreter.interpret(message, context),
                    timeout=self.max_interpretation_time
                )
                for interpreter in self.interpreters
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter successful interpretations
            valid_interpretations = []
            for i, result in enumerate(results):
                if isinstance(result, Interpretation) and result.intent != "error":
                    valid_interpretations.append(result)
                elif isinstance(result, Exception):
                    logger.error(f"Interpreter {i} failed: {result}")
            
            if not valid_interpretations:
                # All interpreters failed
                fallback = Interpretation(
                    intent="error",
                    entities={"error": "all_interpreters_failed"},
                    confidence=0.0,
                    response="I'm having trouble understanding right now. Please try rephrasing your request."
                )
                return ConsensusResult(
                    final_interpretation=fallback,
                    individual_results=[],
                    consensus_confidence=0.0,
                    disagreement_areas=["total_failure"],
                    processing_time=time.time() - start_time
                )
            
            # Calculate consensus
            consensus = self._calculate_consensus(valid_interpretations)
            disagreements = self._identify_disagreements(valid_interpretations)
            
            processing_time = time.time() - start_time
            
            return ConsensusResult(
                final_interpretation=consensus,
                individual_results=valid_interpretations,
                consensus_confidence=consensus.confidence,
                disagreement_areas=disagreements,
                processing_time=processing_time
            )
            
        except asyncio.TimeoutError:
            logger.error("LLM interpretation timed out")
            timeout_result = Interpretation(
                intent="error",
                entities={"error": "interpretation_timeout"},
                confidence=0.0,
                response="Processing took too long. Please try a simpler request."
            )
            return ConsensusResult(
                final_interpretation=timeout_result,
                individual_results=[],
                consensus_confidence=0.0,
                disagreement_areas=["timeout"],
                processing_time=time.time() - start_time
            )
        except Exception as e:
            logger.error(f"Consensus interpretation failed: {e}")
            error_result = Interpretation(
                intent="error",
                entities={"error": str(e)},
                confidence=0.0,
                response="I encountered an error while processing your request."
            )
            return ConsensusResult(
                final_interpretation=error_result,
                individual_results=[],
                consensus_confidence=0.0,
                disagreement_areas=["processing_error"],
                processing_time=time.time() - start_time
            )
    
    def _calculate_consensus(self, interpretations: List[Interpretation]) -> Interpretation:
        """Calculate consensus interpretation from multiple LLM results."""
        if len(interpretations) == 1:
            return interpretations[0]
        
        # Count intent votes
        intent_votes = Counter(interp.intent for interp in interpretations)
        most_common_intent, intent_count = intent_votes.most_common(1)[0]
        
        # Calculate consensus confidence
        intent_consensus = intent_count / len(interpretations)
        
        # Get interpretations with consensus intent
        consensus_interpretations = [
            interp for interp in interpretations 
            if interp.intent == most_common_intent
        ]
        
        # Merge entities from consensus interpretations
        merged_entities = {}
        for interp in consensus_interpretations:
            for key, value in interp.entities.items():
                if key not in merged_entities:
                    merged_entities[key] = []
                merged_entities[key].append(value)
        
        # Resolve entity conflicts by taking most common values
        final_entities = {}
        for key, values in merged_entities.items():
            if isinstance(values[0], dict):
                # For dict values, merge them
                final_entities[key] = {}
                for value_dict in values:
                    final_entities[key].update(value_dict)
            else:
                # For simple values, take most common
                value_counts = Counter(str(v) for v in values)
                final_entities[key] = value_counts.most_common(1)[0][0]
        
        # Select best response from consensus interpretations
        responses = [interp.response for interp in consensus_interpretations if interp.response]
        best_response = responses[0] if responses else None
        
        # Calculate overall confidence
        avg_confidence = sum(interp.confidence for interp in consensus_interpretations) / len(consensus_interpretations)
        final_confidence = intent_consensus * avg_confidence
        
        # Add consensus metadata
        final_entities["consensus_metadata"] = {
            "intent_agreement": intent_consensus,
            "participating_llms": len(interpretations),
            "consensus_llms": len(consensus_interpretations)
        }
        
        return Interpretation(
            intent=most_common_intent,
            entities=final_entities,
            confidence=final_confidence,
            response=best_response,
            reasoning=f"Consensus from {len(consensus_interpretations)}/{len(interpretations)} LLMs"
        )
    
    def _identify_disagreements(self, interpretations: List[Interpretation]) -> List[str]:
        """Identify areas where LLMs disagreed."""
        disagreements = []
        
        # Check intent disagreement
        intents = [interp.intent for interp in interpretations]
        if len(set(intents)) > 1:
            disagreements.append("intent")
        
        # Check entity disagreement
        all_entity_keys = set()
        for interp in interpretations:
            all_entity_keys.update(interp.entities.keys())
        
        for key in all_entity_keys:
            values = []
            for interp in interpretations:
                if key in interp.entities:
                    values.append(str(interp.entities[key]))
            
            if len(set(values)) > 1:
                disagreements.append(f"entity_{key}")
        
        # Check confidence disagreement
        confidences = [interp.confidence for interp in interpretations]
        if max(confidences) - min(confidences) > 0.3:
            disagreements.append("confidence")
        
        return disagreements
    
    def _simple_pattern_matching(self, message: str, context: Dict[str, Any]) -> Interpretation:
        """Fallback pattern matching when LLMs are unavailable."""
        message_lower = message.lower()
        
        # Simple intent detection
        if any(word in message_lower for word in ['status', 'health', 'how']):
            intent = "status_inquiry"
            response = "I'll check the system status for you."
        elif any(word in message_lower for word in ['create', 'build', 'make', 'generate']):
            intent = "task_creation"
            response = "I'll help you create that."
        elif any(word in message_lower for word in ['hello', 'hi', 'hey']):
            intent = "greeting"
            response = "Hello! How can I help you today?"
        elif any(word in message_lower for word in ['help', 'what', 'how']):
            intent = "help_request"
            response = "I'm here to help! What would you like to know?"
        elif message_lower.startswith('/'):
            intent = "terminal_command"
            response = "I'll execute that command for you."
        else:
            intent = "clarification_needed"
            response = "Could you clarify what you'd like me to do?"
        
        return Interpretation(
            intent=intent,
            entities={"message": message, "method": "pattern_matching"},
            confidence=0.3,  # Low confidence for pattern matching
            response=response,
            reasoning="Simple pattern matching (no LLMs available)"
        )

# Factory function for easy initialization
def create_reasoning_engine(config: Dict[str, Any]) -> MultiLLMReasoningEngine:
    """Create and configure the reasoning engine."""
    llm_configs = config.get("llms", [])
    
    # Default configuration if none provided
    if not llm_configs:
        logger.warning("No LLM configs provided, using default configuration")
        llm_configs = [
            {
                "name": "gpt4",
                "model": "gpt-4o",
                "temperature": 0.3,
                "timeout": 8,
                "api_key": "your-openai-key"  # Will need to be configured
            }
        ]
    
    engine = MultiLLMReasoningEngine(llm_configs)
    
    # Configure thresholds
    engine.consensus_threshold = config.get("consensus_threshold", 0.8)
    engine.clarification_threshold = config.get("clarification_threshold", 0.5)
    engine.max_interpretation_time = config.get("max_interpretation_time", 10)
    
    return engine
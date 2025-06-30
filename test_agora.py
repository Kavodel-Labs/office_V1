"""
Test script for Agora Consensus Engine
Tests all 4 LLM integrations and consensus mechanisms
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import Dict, Any

# Import Agora components
from core.agora import AgoraOrchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

async def test_agora_consensus():
    """Test Agora consensus with multiple task types"""
    
    # Configure with API keys from environment
    config = {
        'llm_config': {
            'openai_api_key': os.getenv('OPENAI_API_KEY'),
            'anthropic_api_key': os.getenv('ANTHROPIC_API_KEY'), 
            'google_api_key': os.getenv('GOOGLE_API_KEY'),
            'perplexity_api_key': os.getenv('PERPLEXITY_API_KEY')
        },
        'budget_limits': {
            'daily_limit': 5.0,
            'session_limit': 1.0,
            'loop_limit': 0.25
        },
        'quality_threshold': 0.7  # Lower for testing
    }
    
    # Initialize Agora
    agora = AgoraOrchestrator(config)
    
    # Test cases
    test_cases = [
        {
            'name': 'Simple Code Generation',
            'task': {
                'id': 'test_code_1',
                'type': 'code_generation',
                'prompt': 'Write a Python function to calculate fibonacci numbers efficiently',
                'complexity': 'medium',
                'max_loops': 1
            }
        },
        {
            'name': 'System Design Question',
            'task': {
                'id': 'test_design_1', 
                'type': 'system_design',
                'prompt': 'Design a microservices architecture for a real-time chat application',
                'complexity': 'high',
                'max_loops': 1
            }
        },
        {
            'name': 'Research Task',
            'task': {
                'id': 'test_research_1',
                'type': 'research_request',
                'prompt': 'Explain the latest developments in transformer architecture optimizations',
                'complexity': 'medium',
                'max_loops': 1
            }
        },
        {
            'name': 'Bug Investigation',
            'task': {
                'id': 'test_bug_1',
                'type': 'bug_investigation', 
                'prompt': 'Analyze why a React component is causing memory leaks and suggest fixes',
                'complexity': 'high',
                'max_loops': 1
            }
        }
    ]
    
    results = []
    
    for test_case in test_cases:
        logger.info(f"\n{'='*60}")
        logger.info(f"🧪 Running test: {test_case['name']}")
        logger.info(f"{'='*60}")
        
        try:
            start_time = datetime.now()
            
            # Invoke Agora consensus
            result = await agora.invoke(test_case['task'], 'test_runner')
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Store results
            test_result = {
                'test_name': test_case['name'],
                'task_type': test_case['task']['type'],
                'success': True,
                'duration_seconds': duration,
                'confidence': result.get('confidence', 0),
                'total_cost': result.get('total_cost', 0),
                'method': result.get('method', 'unknown'),
                'content_length': len(result.get('content', '')),
                'reasoning_trace': result.get('reasoning_trace', {})
            }
            
            results.append(test_result)
            
            # Log summary
            logger.info(f"✅ Test '{test_case['name']}' completed:")
            logger.info(f"   - Duration: {duration:.2f}s")
            logger.info(f"   - Confidence: {result.get('confidence', 0):.2f}")
            logger.info(f"   - Cost: ${result.get('total_cost', 0):.3f}")
            logger.info(f"   - Method: {result.get('method', 'unknown')}")
            logger.info(f"   - Content length: {len(result.get('content', ''))} chars")
            
            # Show reasoning trace summary
            trace = result.get('reasoning_trace', {})
            if trace:
                logger.info(f"   - Specialists: {trace.get('specialist_responses', [])}")
                logger.info(f"   - Loops: {trace.get('loops', 0)}")
                
                # Show merged result summary
                merged = trace.get('merged_result', {})
                if merged:
                    logger.info(f"   - Consensus points: {len(merged.get('consensus', []))}")
                    logger.info(f"   - Conflicts: {len(merged.get('conflicts', []))}")
                    logger.info(f"   - Agreement ratio: {merged.get('agreement_ratio', 0):.2f}")
                    
        except Exception as e:
            logger.error(f"❌ Test '{test_case['name']}' failed: {e}")
            
            test_result = {
                'test_name': test_case['name'],
                'task_type': test_case['task']['type'],
                'success': False,
                'error': str(e),
                'duration_seconds': 0,
                'confidence': 0,
                'total_cost': 0
            }
            
            results.append(test_result)
    
    # Generate final report
    logger.info(f"\n{'='*60}")
    logger.info("📊 AGORA CONSENSUS TEST REPORT")
    logger.info(f"{'='*60}")
    
    successful_tests = [r for r in results if r['success']]
    failed_tests = [r for r in results if not r['success']]
    
    logger.info(f"✅ Successful tests: {len(successful_tests)}/{len(results)}")
    logger.info(f"❌ Failed tests: {len(failed_tests)}")
    
    if successful_tests:
        total_cost = sum(r['total_cost'] for r in successful_tests)
        avg_confidence = sum(r['confidence'] for r in successful_tests) / len(successful_tests)
        avg_duration = sum(r['duration_seconds'] for r in successful_tests) / len(successful_tests)
        
        logger.info(f"💰 Total cost: ${total_cost:.3f}")
        logger.info(f"🎯 Average confidence: {avg_confidence:.2f}")
        logger.info(f"⏱️ Average duration: {avg_duration:.2f}s")
        
        # Success rate by task type
        task_types = {}
        for result in results:
            task_type = result['task_type']
            if task_type not in task_types:
                task_types[task_type] = {'total': 0, 'success': 0}
            task_types[task_type]['total'] += 1
            if result['success']:
                task_types[task_type]['success'] += 1
        
        logger.info("\n📈 Success rate by task type:")
        for task_type, stats in task_types.items():
            success_rate = stats['success'] / stats['total'] * 100
            logger.info(f"   - {task_type}: {success_rate:.1f}% ({stats['success']}/{stats['total']})")
    
    if failed_tests:
        logger.info("\n❌ Failed test details:")
        for result in failed_tests:
            logger.info(f"   - {result['test_name']}: {result.get('error', 'Unknown error')}")
    
    logger.info(f"\n{'='*60}")
    logger.info("🎉 Agora Consensus Engine testing complete!")
    logger.info(f"{'='*60}")
    
    return results

async def test_individual_components():
    """Test individual Agora components"""
    logger.info("\n🔧 Testing individual Agora components...")
    
    # Test Role Assignment
    from core.agora.role_assignment import RoleAssignment
    
    logger.info("Testing Role Assignment...")
    specialists = RoleAssignment.get_specialists("code_generation", "high")
    logger.info(f"Code generation specialists: {specialists}")
    
    task_type = RoleAssignment.recommend_task_type("Write a Python function for sorting")
    logger.info(f"Recommended task type: {task_type}")
    
    # Test Cost Controller
    from core.agora.cost_controller import CostController
    
    logger.info("Testing Cost Controller...")
    cost_controller = CostController({
        'daily_limit': 1.0,
        'session_limit': 0.5
    })
    
    metrics = await cost_controller.get_budget_metrics()
    logger.info(f"Budget metrics: ${metrics.remaining_budget:.3f} remaining")
    
    logger.info("✅ Component tests complete")

if __name__ == "__main__":
    # Check for required environment variables
    required_keys = ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'GOOGLE_API_KEY', 'PERPLEXITY_API_KEY']
    missing_keys = [key for key in required_keys if not os.getenv(key)]
    
    if missing_keys:
        logger.error(f"Missing required environment variables: {missing_keys}")
        logger.error("Please set these API keys before running the test")
        exit(1)
    
    # Run tests
    asyncio.run(test_individual_components())
    asyncio.run(test_agora_consensus())
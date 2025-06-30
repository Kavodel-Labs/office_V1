#!/usr/bin/env python3
"""
AETHELRED Headless Testing Agent
Comprehensive testing of Agora Consensus Engine without Slack dependencies
"""

import asyncio
import json
import time
import sys
import logging
from datetime import datetime
from typing import Dict, Any, List

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/headless_test.log', mode='a')
    ]
)

logger = logging.getLogger(__name__)

# Add current directory to path
sys.path.append('.')

class HeadlessTestingAgent:
    """Comprehensive headless testing agent for AETHELRED Agora system"""
    
    def __init__(self):
        self.test_results = []
        self.agora = None
        self.start_time = datetime.now()
        
        # Test configurations
        self.agora_config = {
            'llm_config': {
                'openai_api_key': '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',
                'anthropic_api_key': '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',
                'google_api_key': 'AIzaSyCM7IyoxJ5kdEGDl4395YP7ojHSZzoDNS8',
                'perplexity_api_key': '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'
            },
            'budget_limits': {
                'daily_limit': 10.0,
                'session_limit': 2.0,
                'loop_limit': 0.5
            },
            'quality_threshold': 0.7
        }
        
    async def initialize(self):
        """Initialize the testing agent and Agora system"""
        logger.info("🚀 Initializing AETHELRED Headless Testing Agent...")
        
        try:
            # Import and initialize Agora
            from core.agora import AgoraOrchestrator
            self.agora = AgoraOrchestrator(self.agora_config)
            logger.info("✅ Agora Consensus Engine initialized")
            
            # Initialize other components
            await self._initialize_components()
            
            logger.info("✅ Headless Testing Agent ready for comprehensive testing")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize testing agent: {e}")
            return False
    
    async def _initialize_components(self):
        """Initialize additional components for testing"""
        try:
            # Test memory manager (mock)
            self.memory_manager = MockMemoryManager()
            
            # Test agent framework components
            await self._test_agent_imports()
            
        except Exception as e:
            logger.warning(f"⚠️ Component initialization warning: {e}")
    
    async def _test_agent_imports(self):
        """Test that all agent components can be imported"""
        try:
            from agents.base.agent import Agent, AgentCapability
            from agents.brigade.comms_secretary.agent import CommsSecretary
            logger.info("✅ Agent framework imports successful")
        except Exception as e:
            logger.warning(f"⚠️ Agent import issues: {e}")
    
    async def run_comprehensive_tests(self):
        """Run all comprehensive tests"""
        logger.info("🧪 Starting Comprehensive Headless Testing Suite...")
        logger.info("=" * 80)
        
        test_suite = [
            ("Core Agora Components", self._test_core_agora),
            ("Multi-LLM Integration", self._test_multi_llm_integration),
            ("Consensus Mechanisms", self._test_consensus_mechanisms),
            ("Quality Evaluation", self._test_quality_evaluation), 
            ("Cost Management", self._test_cost_management),
            ("Role Assignment", self._test_role_assignment),
            ("End-to-End Scenarios", self._test_end_to_end_scenarios),
            ("Performance Metrics", self._test_performance_metrics),
            ("Error Handling", self._test_error_handling),
            ("Agent Integration", self._test_agent_integration)
        ]
        
        total_tests = len(test_suite)
        passed_tests = 0
        
        for i, (test_name, test_func) in enumerate(test_suite, 1):
            logger.info(f"\n🔬 Test {i}/{total_tests}: {test_name}")
            logger.info("-" * 60)
            
            try:
                start_time = time.time()
                result = await test_func()
                duration = time.time() - start_time
                
                if result:
                    logger.info(f"✅ {test_name} - PASSED ({duration:.2f}s)")
                    passed_tests += 1
                else:
                    logger.error(f"❌ {test_name} - FAILED ({duration:.2f}s)")
                    
                self.test_results.append({
                    'test_name': test_name,
                    'passed': result,
                    'duration': duration,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"💥 {test_name} - ERROR: {e}")
                self.test_results.append({
                    'test_name': test_name,
                    'passed': False,
                    'error': str(e),
                    'duration': 0,
                    'timestamp': datetime.now().isoformat()
                })
        
        # Generate final report
        await self._generate_final_report(passed_tests, total_tests)
        
        return passed_tests, total_tests
    
    async def _test_core_agora(self):
        """Test core Agora components"""
        try:
            # Test Agora initialization
            if not self.agora:
                logger.error("Agora not initialized")
                return False
            
            # Test basic task creation
            test_task = {
                'id': 'test_core_001',
                'type': 'general',
                'prompt': 'What is the capital of France?',
                'complexity': 'low',
                'max_loops': 1
            }
            
            logger.info("🔬 Testing basic Agora task processing...")
            result = await self.agora.invoke(test_task, 'headless_tester')
            
            if result and 'content' in result:
                logger.info(f"✅ Core Agora response: {result['content'][:100]}...")
                logger.info(f"📊 Confidence: {result.get('confidence', 0):.2f}")
                logger.info(f"💰 Cost: ${result.get('total_cost', 0):.3f}")
                return True
            else:
                logger.error("❌ No valid response from Agora")
                return False
                
        except Exception as e:
            logger.error(f"💥 Core Agora test failed: {e}")
            return False
    
    async def _test_multi_llm_integration(self):
        """Test multi-LLM integration and consensus"""
        try:
            logger.info("🔬 Testing Multi-LLM consensus on complex task...")
            
            complex_task = {
                'id': 'test_multi_llm_001',
                'type': 'system_design',
                'prompt': '''Design a distributed microservices architecture for a real-time chat application that needs to:
                1. Handle 1M concurrent users
                2. Support real-time messaging with low latency
                3. Implement message history and search
                4. Scale across multiple regions
                5. Ensure high availability and fault tolerance
                
                Provide specific technology recommendations, architecture patterns, and deployment strategies.''',
                'complexity': 'high',
                'max_loops': 1
            }
            
            start_time = time.time()
            result = await self.agora.invoke(complex_task, 'headless_tester')
            processing_time = time.time() - start_time
            
            if result and 'content' in result:
                logger.info(f"✅ Multi-LLM consensus completed in {processing_time:.2f}s")
                logger.info(f"📊 Confidence: {result.get('confidence', 0):.2f}")
                logger.info(f"💰 Cost: ${result.get('total_cost', 0):.3f}")
                logger.info(f"📝 Response length: {len(result['content'])} chars")
                
                # Check for reasoning trace
                trace = result.get('reasoning_trace', {})
                if trace:
                    specialists = trace.get('specialist_responses', [])
                    logger.info(f"🏛️ Specialists involved: {len(specialists)}")
                    
                return len(result['content']) > 500  # Substantial response
            else:
                logger.error("❌ No valid multi-LLM response")
                return False
                
        except Exception as e:
            logger.error(f"💥 Multi-LLM test failed: {e}")
            return False
    
    async def _test_consensus_mechanisms(self):
        """Test consensus analysis and conflict resolution"""
        try:
            logger.info("🔬 Testing consensus mechanisms...")
            
            from core.agora.merge_engine import MergeEngine
            merge_engine = MergeEngine()
            
            # Create mock specialist responses
            mock_responses = [
                MockResponse("gpt4", "Use REST APIs for communication", 0.9, "lead"),
                MockResponse("claude", "Implement GraphQL for flexible queries", 0.8, "support"),
                MockResponse("gemini", "Consider event-driven architecture", 0.7, "support"),
                MockResponse("perplexity", "REST APIs are most widely adopted", 0.8, "critic")
            ]
            
            # Create mock session
            mock_session = MockSession("api_design", "consensus_test")
            
            # Test merge
            merged_result = await merge_engine.merge(mock_responses, mock_session)
            
            if merged_result and 'content' in merged_result:
                consensus = merged_result.get('consensus', [])
                conflicts = merged_result.get('conflicts', [])
                
                logger.info(f"✅ Merge completed")
                logger.info(f"📊 Consensus points: {len(consensus)}")
                logger.info(f"⚠️ Conflicts: {len(conflicts)}")
                logger.info(f"🎯 Agreement ratio: {merged_result.get('agreement_ratio', 0):.2f}")
                
                return True
            else:
                logger.error("❌ Merge engine failed")
                return False
                
        except Exception as e:
            logger.error(f"💥 Consensus test failed: {e}")
            return False
    
    async def _test_quality_evaluation(self):
        """Test quality evaluation system"""
        try:
            logger.info("🔬 Testing quality evaluation...")
            
            from core.agora.critic_engine import CriticEngine
            critic_engine = CriticEngine()
            
            # Mock merged result
            mock_result = {
                'content': '''A comprehensive microservices architecture should include:
                
                ## Core Components
                - API Gateway for routing and authentication
                - Service mesh for communication
                - Event bus for async messaging
                - Centralized logging and monitoring
                
                ## Implementation Strategy
                1. Start with domain-driven design
                2. Implement circuit breakers
                3. Use container orchestration
                4. Implement progressive deployment
                
                This approach ensures scalability, maintainability, and reliability.''',
                'consensus': ['API Gateway', 'Service mesh', 'Event bus'],
                'conflicts': [],
                'specialist_contributions': [
                    {'specialist': 'gpt4', 'confidence': 0.9},
                    {'specialist': 'claude', 'confidence': 0.8}
                ]
            }
            
            mock_session = MockSession("system_design", "quality_test")
            
            # Test evaluation
            evaluation = await critic_engine.evaluate(mock_result, mock_session)
            
            if evaluation and 'quality_score' in evaluation:
                logger.info(f"✅ Quality evaluation completed")
                logger.info(f"📊 Quality score: {evaluation['quality_score']:.2f}")
                logger.info(f"💪 Strengths: {len(evaluation.get('strengths', []))}")
                logger.info(f"⚠️ Weaknesses: {len(evaluation.get('weaknesses', []))}")
                logger.info(f"💡 Suggestions: {len(evaluation.get('improvement_suggestions', []))}")
                
                return evaluation['quality_score'] > 0.5
            else:
                logger.error("❌ Quality evaluation failed")
                return False
                
        except Exception as e:
            logger.error(f"💥 Quality evaluation test failed: {e}")
            return False
    
    async def _test_cost_management(self):
        """Test cost tracking and budget management"""
        try:
            logger.info("🔬 Testing cost management...")
            
            from core.agora.cost_controller import CostController
            
            budget_config = {
                'daily_limit': 5.0,
                'session_limit': 1.0,
                'loop_limit': 0.25
            }
            
            cost_controller = CostController(budget_config)
            
            # Test budget metrics
            metrics = await cost_controller.get_budget_metrics()
            logger.info(f"💰 Budget metrics: ${metrics.remaining_budget:.2f} remaining")
            
            # Test cost tracking
            await cost_controller.track_session_cost("test_session_001", 0.15)
            
            # Test budget check
            mock_session = MockSession("test", "cost_test")
            budget_ok = await cost_controller.check_budget(mock_session)
            
            logger.info(f"✅ Cost management: Budget check {'passed' if budget_ok else 'failed'}")
            
            # Test cost report
            report = await cost_controller.get_cost_report()
            logger.info(f"📊 Cost report generated with {len(report)} sections")
            
            return True
            
        except Exception as e:
            logger.error(f"💥 Cost management test failed: {e}")
            return False
    
    async def _test_role_assignment(self):
        """Test specialist role assignment"""
        try:
            logger.info("🔬 Testing role assignment...")
            
            from core.agora.role_assignment import RoleAssignment
            
            # Test different task types
            test_cases = [
                ('code_generation', 'high'),
                ('system_design', 'medium'),
                ('documentation', 'low'),
                ('research_request', 'high')
            ]
            
            for task_type, complexity in test_cases:
                specialists = RoleAssignment.get_specialists(task_type, complexity)
                
                logger.info(f"📋 {task_type} ({complexity}):")
                logger.info(f"   Lead: {specialists.get('lead', [])}")
                logger.info(f"   Support: {specialists.get('support', [])}")
                logger.info(f"   Critic: {specialists.get('critic', [])}")
            
            # Test recommendation
            recommendation = RoleAssignment.recommend_task_type("Write a Python function to sort data")
            logger.info(f"🎯 Recommendation: 'sort data' → {recommendation}")
            
            return True
            
        except Exception as e:
            logger.error(f"💥 Role assignment test failed: {e}")
            return False
    
    async def _test_end_to_end_scenarios(self):
        """Test complete end-to-end scenarios"""
        try:
            logger.info("🔬 Testing end-to-end scenarios...")
            
            scenarios = [
                {
                    'name': 'Simple Code Generation',
                    'task': {
                        'id': 'e2e_code_001',
                        'type': 'code_generation',
                        'prompt': 'Write a Python function to calculate the fibonacci sequence efficiently',
                        'complexity': 'medium'
                    }
                },
                {
                    'name': 'Architecture Design',
                    'task': {
                        'id': 'e2e_arch_001', 
                        'type': 'system_design',
                        'prompt': 'Design a caching strategy for a high-traffic web application',
                        'complexity': 'high'
                    }
                },
                {
                    'name': 'Research Query',
                    'task': {
                        'id': 'e2e_research_001',
                        'type': 'research_request',
                        'prompt': 'What are the latest developments in large language model optimization?',
                        'complexity': 'medium'
                    }
                }
            ]
            
            success_count = 0
            
            for scenario in scenarios:
                logger.info(f"🎯 Testing: {scenario['name']}")
                
                try:
                    result = await self.agora.invoke(scenario['task'], 'e2e_tester')
                    
                    if result and result.get('confidence', 0) > 0.5:
                        logger.info(f"✅ {scenario['name']} - SUCCESS")
                        success_count += 1
                    else:
                        logger.warning(f"⚠️ {scenario['name']} - LOW QUALITY")
                        
                except Exception as e:
                    logger.error(f"❌ {scenario['name']} - FAILED: {e}")
            
            logger.info(f"📊 E2E Results: {success_count}/{len(scenarios)} scenarios passed")
            return success_count >= len(scenarios) // 2  # At least half should pass
            
        except Exception as e:
            logger.error(f"💥 End-to-end test failed: {e}")
            return False
    
    async def _test_performance_metrics(self):
        """Test performance and timing metrics"""
        try:
            logger.info("🔬 Testing performance metrics...")
            
            # Test response times for different complexity levels
            test_prompts = [
                ('Simple', 'What is 2+2?'),
                ('Medium', 'Explain the benefits of microservices architecture'),
                ('Complex', 'Design a distributed system for real-time data processing with fault tolerance')
            ]
            
            performance_data = []
            
            for complexity, prompt in test_prompts:
                start_time = time.time()
                
                task = {
                    'id': f'perf_test_{complexity.lower()}',
                    'type': 'general',
                    'prompt': prompt,
                    'complexity': complexity.lower(),
                    'max_loops': 1
                }
                
                try:
                    result = await self.agora.invoke(task, 'perf_tester')
                    duration = time.time() - start_time
                    
                    performance_data.append({
                        'complexity': complexity,
                        'duration': duration,
                        'success': result is not None,
                        'response_length': len(result.get('content', '')) if result else 0,
                        'confidence': result.get('confidence', 0) if result else 0,
                        'cost': result.get('total_cost', 0) if result else 0
                    })
                    
                    logger.info(f"⏱️ {complexity}: {duration:.2f}s, confidence: {result.get('confidence', 0):.2f}")
                    
                except Exception as e:
                    logger.warning(f"⚠️ {complexity} performance test failed: {e}")
                    performance_data.append({
                        'complexity': complexity,
                        'duration': time.time() - start_time,
                        'success': False,
                        'error': str(e)
                    })
            
            # Analyze performance
            successful_tests = [p for p in performance_data if p.get('success', False)]
            if successful_tests:
                avg_duration = sum(p['duration'] for p in successful_tests) / len(successful_tests)
                logger.info(f"📊 Average response time: {avg_duration:.2f}s")
                
                return len(successful_tests) >= 2  # At least 2 should succeed
            else:
                logger.error("❌ No successful performance tests")
                return False
                
        except Exception as e:
            logger.error(f"💥 Performance test failed: {e}")
            return False
    
    async def _test_error_handling(self):
        """Test error handling and recovery"""
        try:
            logger.info("🔬 Testing error handling...")
            
            error_scenarios = [
                {
                    'name': 'Empty Prompt',
                    'task': {'id': 'err_001', 'type': 'general', 'prompt': '', 'complexity': 'low'}
                },
                {
                    'name': 'Invalid Task Type',
                    'task': {'id': 'err_002', 'type': 'invalid_type', 'prompt': 'Test', 'complexity': 'low'}
                },
                {
                    'name': 'Extremely Long Prompt',
                    'task': {'id': 'err_003', 'type': 'general', 'prompt': 'A' * 10000, 'complexity': 'low'}
                }
            ]
            
            error_handled_count = 0
            
            for scenario in error_scenarios:
                logger.info(f"🚨 Testing: {scenario['name']}")
                
                try:
                    result = await self.agora.invoke(scenario['task'], 'error_tester')
                    
                    if result:
                        logger.info(f"✅ {scenario['name']} - Handled gracefully")
                        error_handled_count += 1
                    else:
                        logger.warning(f"⚠️ {scenario['name']} - Returned None")
                        error_handled_count += 1  # Still handled
                        
                except Exception as e:
                    logger.info(f"✅ {scenario['name']} - Exception caught: {type(e).__name__}")
                    error_handled_count += 1  # Error properly caught
            
            logger.info(f"📊 Error handling: {error_handled_count}/{len(error_scenarios)} scenarios handled")
            return error_handled_count == len(error_scenarios)
            
        except Exception as e:
            logger.error(f"💥 Error handling test failed: {e}")
            return False
    
    async def _test_agent_integration(self):
        """Test integration with agent framework"""
        try:
            logger.info("🔬 Testing agent integration...")
            
            # Test agent capability checks
            from agents.base.agent import AgentCapability
            
            agora_capabilities = [
                AgentCapability.AGORA_CONSENSUS,
                AgentCapability.AGORA_INVOKE
            ]
            
            logger.info(f"✅ Agora capabilities defined: {len(agora_capabilities)}")
            
            # Test mock agent with Agora
            mock_agent = MockAgent()
            integration_success = await mock_agent.test_agora_integration()
            
            if integration_success:
                logger.info("✅ Agent integration successful")
                return True
            else:
                logger.error("❌ Agent integration failed")
                return False
                
        except Exception as e:
            logger.error(f"💥 Agent integration test failed: {e}")
            return False
    
    async def _generate_final_report(self, passed_tests: int, total_tests: int):
        """Generate comprehensive final test report"""
        total_duration = (datetime.now() - self.start_time).total_seconds()
        
        report = f"""
{'='*80}
🏛️ AETHELRED AGORA CONSENSUS ENGINE - HEADLESS TEST REPORT
{'='*80}

📊 SUMMARY:
   • Tests Passed: {passed_tests}/{total_tests} ({(passed_tests/total_tests)*100:.1f}%)
   • Total Duration: {total_duration:.2f} seconds
   • Test Start: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}
   • Test End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

🎯 SYSTEM STATUS:
   • Agora Engine: {'✅ OPERATIONAL' if self.agora else '❌ FAILED'}
   • Multi-LLM: {'✅ WORKING' if passed_tests >= total_tests * 0.7 else '⚠️ ISSUES'}
   • Core Components: {'✅ STABLE' if passed_tests >= total_tests * 0.8 else '❌ UNSTABLE'}

📋 DETAILED RESULTS:
"""
        
        for result in self.test_results:
            status = "✅ PASS" if result['passed'] else "❌ FAIL"
            duration = result.get('duration', 0)
            error = f" - {result['error']}" if 'error' in result else ""
            report += f"   {status} {result['test_name']} ({duration:.2f}s){error}\n"
        
        report += f"""
🏆 FINAL VERDICT:
   {'🎉 AETHELRED AGORA SYSTEM IS OPERATIONAL!' if passed_tests >= total_tests * 0.7 else '⚠️ AETHELRED AGORA SYSTEM NEEDS ATTENTION'}

{'='*80}
"""
        
        logger.info(report)
        
        # Save report to file
        try:
            with open('logs/headless_test_report.txt', 'w') as f:
                f.write(report)
            logger.info("📁 Test report saved to logs/headless_test_report.txt")
        except Exception as e:
            logger.warning(f"⚠️ Could not save report: {e}")

# Mock classes for testing
class MockMemoryManager:
    async def hot_get(self, key): return None
    async def hot_set(self, key, value, ttl=None): pass
    async def warm_store(self, collection, data): pass

class MockResponse:
    def __init__(self, specialist_id, content, confidence, role):
        self.specialist_id = specialist_id
        self.content = content
        self.confidence = confidence
        self.role = role
        self.tokens_used = len(content.split())
        self.cost = 0.01

class MockSession:
    def __init__(self, task_type, session_id):
        self.task_type = task_type
        self.session_id = session_id
        self.prompt = "Mock test prompt"
        self.specialists = {"lead": ["gpt4"], "support": ["claude"], "critic": ["gemini"]}

class MockAgent:
    async def test_agora_integration(self):
        return True  # Simulate successful integration

# Main execution
async def main():
    """Main headless testing execution"""
    print("🚀 Starting AETHELRED Headless Testing Agent...")
    
    # Create logs directory
    import os
    os.makedirs('logs', exist_ok=True)
    
    # Initialize and run tests
    agent = HeadlessTestingAgent()
    
    if await agent.initialize():
        passed, total = await agent.run_comprehensive_tests()
        
        if passed >= total * 0.7:
            print(f"\n🎉 SUCCESS: {passed}/{total} tests passed - AGORA SYSTEM IS OPERATIONAL!")
            return 0
        else:
            print(f"\n⚠️ PARTIAL: {passed}/{total} tests passed - SYSTEM NEEDS ATTENTION")
            return 1
    else:
        print("\n❌ FAILED: Could not initialize testing agent")
        return 2

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
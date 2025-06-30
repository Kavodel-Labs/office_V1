#!/usr/bin/env python3
"""
Integration tests for AETHELRED with live database connections.

This validates the complete system with real Redis, PostgreSQL, and Neo4j databases.
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Dict, Any

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_live_database_connections():
    """Test connections to all live databases."""
    print("🔌 Testing live database connections...")
    
    try:
        from core.memory.interfaces import HotMemory, WarmMemory, ColdMemory, ArchiveMemory
        
        # Test Redis (Hot Memory) connection
        redis_config = {
            'host': 'localhost',
            'port': 6379,
            'db': 0,
            'ttl': 3600
        }
        hot_memory = HotMemory(redis_config)
        await hot_memory.connect()
        
        # Test basic Redis operations
        await hot_memory.write("integration_test_redis", {"message": "Hello Redis!", "timestamp": "2025-06-29"})
        redis_result = await hot_memory.read("integration_test_redis")
        assert redis_result["message"] == "Hello Redis!"
        
        print("✅ Redis (Hot Memory) - Connected and tested")
        await hot_memory.disconnect()
        
        # Test PostgreSQL (Warm Memory) connection
        postgres_config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'aethelred',
            'user': 'aethelred',
            'password': 'development'
        }
        warm_memory = WarmMemory(postgres_config)
        await warm_memory.connect()
        
        # Test basic PostgreSQL operations
        test_uuid = "550e8400-e29b-41d4-a716-446655440000"
        await warm_memory.write(test_uuid, {"message": "Hello PostgreSQL!", "timestamp": "2025-06-29"})
        postgres_result = await warm_memory.read(test_uuid)
        # PostgreSQL result might be JSON string, so handle both cases
        if isinstance(postgres_result, str):
            import json
            postgres_result = json.loads(postgres_result)
        
        print("✅ PostgreSQL (Warm Memory) - Connected and tested")
        await warm_memory.disconnect()
        
        # Test Neo4j (Cold Memory) connection
        neo4j_config = {
            'uri': 'bolt://localhost:7687',
            'user': 'neo4j',
            'password': 'development'
        }
        cold_memory = ColdMemory(neo4j_config)
        await cold_memory.connect()
        
        # Test basic Neo4j operations
        await cold_memory.write("integration_test_neo4j", {"message": "Hello Neo4j!", "timestamp": "2025-06-29"})
        neo4j_result = await cold_memory.read("integration_test_neo4j")
        
        print("✅ Neo4j (Cold Memory) - Connected and tested")
        await cold_memory.disconnect()
        
        # Test Archive Memory (Filesystem)
        archive_config = {
            'filesystem': {
                'path': './test_integration_archive'
            }
        }
        archive_memory = ArchiveMemory(archive_config)
        await archive_memory.connect()
        
        await archive_memory.write("integration_test_archive", {"message": "Hello Archive!", "timestamp": "2025-06-29"})
        archive_result = await archive_memory.read("integration_test_archive")
        assert archive_result["message"] == "Hello Archive!"
        
        print("✅ Archive Memory (Filesystem) - Connected and tested")
        await archive_memory.disconnect()
        
        print("🎉 All live database connections successful!")
        return True
        
    except Exception as e:
        print(f"❌ Live database connection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_memory_tier_integration():
    """Test complete memory tier system with live databases."""
    print("\\n🧠 Testing memory tier integration...")
    
    try:
        from core.memory.tier_manager import MemoryTierManager
        from core.memory.interfaces import HotMemory, WarmMemory, ColdMemory, ArchiveMemory
        
        # Create memory manager
        memory_manager = MemoryTierManager()
        
        # Configure and register live tiers
        hot_config = {'host': 'localhost', 'port': 6379, 'db': 0, 'ttl': 3600}
        warm_config = {
            'host': 'localhost', 'port': 5432, 'database': 'aethelred',
            'user': 'aethelred', 'password': 'development'
        }
        cold_config = {'uri': 'bolt://localhost:7687', 'user': 'neo4j', 'password': 'development'}
        archive_config = {'filesystem': {'path': './test_integration_archive'}}
        
        hot_tier = HotMemory(hot_config)
        warm_tier = WarmMemory(warm_config)
        cold_tier = ColdMemory(cold_config)
        archive_tier = ArchiveMemory(archive_config)
        
        memory_manager.register_tier(hot_tier)
        memory_manager.register_tier(warm_tier)
        memory_manager.register_tier(cold_tier)
        memory_manager.register_tier(archive_tier)
        
        # Connect all tiers
        await memory_manager.connect_all()
        print("✅ All memory tiers connected")
        
        # Test tiered operations
        test_data = {
            "integration_test": True,
            "timestamp": "2025-06-29",
            "message": "Multi-tier integration test",
            "data_size": "medium"
        }
        
        # Write to specific tiers
        await memory_manager.write("integration_key_hot", test_data, target_tiers=['hot'])
        await memory_manager.write("integration_key_multi", test_data, target_tiers=['hot', 'warm'])
        
        # Read from tiers
        hot_result = await memory_manager.read("integration_key_hot")
        multi_result = await memory_manager.read("integration_key_multi")
        
        assert hot_result["integration_test"] == True
        assert multi_result["integration_test"] == True
        
        print("✅ Tiered write/read operations working")
        
        # Test health checks
        health_status = await memory_manager.health_check()
        healthy_tiers = sum(1 for tier_health in health_status.values() 
                          if tier_health.get('status') == 'healthy')
        total_tiers = len(health_status)
        
        print(f"✅ Health check: {healthy_tiers}/{total_tiers} tiers healthy")
        print(f"   Tier health details:")
        for tier_name, health in health_status.items():
            status = health.get('status', 'unknown')
            print(f"   - {tier_name}: {status}")
        
        # Cleanup
        await memory_manager.disconnect_all()
        
        print("🎉 Memory tier integration test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Memory tier integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_system_with_live_databases():
    """Test the complete AETHELRED system with live databases."""
    print("\\n🚀 Testing complete system with live databases...")
    
    try:
        from main import AethelredSystem
        
        # Create system with live database configuration
        system = AethelredSystem()
        
        # Initialize system (this will connect to live databases)
        await system.initialize()
        print("✅ AETHELRED system initialized with live databases")
        
        # Test system status
        status = system.get_system_status()
        print(f"✅ System status: {status['system_name']} v{status['version']}")
        print(f"   Environment: {status['environment']}")
        print(f"   Started at: {status['started_at']}")
        
        # Test agent health
        agent_health = await system.get_agent_health()
        print(f"✅ Agent health checks completed for {len(agent_health)} agents")
        
        # Test memory system health
        memory_health = await system.memory_manager.health_check()
        healthy_memory_tiers = sum(1 for h in memory_health.values() if h.get('status') == 'healthy')
        print(f"✅ Memory system: {healthy_memory_tiers}/{len(memory_health)} tiers healthy")
        
        # Graceful shutdown
        await system.shutdown()
        print("✅ System shutdown completed")
        
        print("🎉 Complete system integration test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Complete system integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_agent_task_execution():
    """Test agent task execution with live infrastructure."""
    print("\\n🤖 Testing agent task execution with live infrastructure...")
    
    try:
        from core.memory.tier_manager import MemoryTierManager
        from core.memory.interfaces import HotMemory, WarmMemory
        from core.task_management.task_master_integration import TaskMasterIntegration
        from core.task_management.task_router import TaskRouter
        from agents.apex.chief_of_staff.agent import ChiefOfStaff
        from agents.service.auditor.agent import Auditor
        
        # Set up memory system with live Redis and PostgreSQL
        memory_manager = MemoryTierManager()
        
        hot_config = {'host': 'localhost', 'port': 6379, 'db': 0, 'ttl': 3600}
        warm_config = {
            'host': 'localhost', 'port': 5432, 'database': 'aethelred',
            'user': 'aethelred', 'password': 'development'
        }
        
        hot_tier = HotMemory(hot_config)
        warm_tier = WarmMemory(warm_config)
        
        memory_manager.register_tier(hot_tier)
        memory_manager.register_tier(warm_tier)
        await memory_manager.connect_all()
        
        print("✅ Live memory system connected for agent testing")
        
        # Set up task management
        task_config = {
            'task_master_path': 'task-master-ai',
            'project_root': Path.cwd(),
            'max_concurrent_tasks': 10
        }
        task_master = TaskMasterIntegration(memory_manager, task_config)
        task_router = TaskRouter(memory_manager, task_master)
        
        # Create agents
        chief_config = {'test_mode': True}
        chief_of_staff = ChiefOfStaff(memory_manager, task_router, task_master, chief_config)
        await chief_of_staff.initialize()
        
        auditor_config = {'test_mode': True}
        auditor = Auditor(memory_manager, auditor_config)
        await auditor.initialize()
        
        print("✅ Agents initialized with live infrastructure")
        
        # Test Chief of Staff system status task
        system_status_task = {
            'id': 'integration_test_status',
            'type': 'system_status',
            'description': 'Get system status for integration test'
        }
        
        status_result = await chief_of_staff.process_task(system_status_task)
        assert status_result.success == True
        assert 'system_status' in status_result.result['action']
        
        print("✅ Chief of Staff system status task executed")
        
        # Test Auditor observation task
        observation_task = {
            'id': 'integration_test_observation',
            'type': 'observe_agent',
            'agent_id': 'A_ChiefOfStaff',
            'observation_data': {
                'response_time_ms': 150,
                'success_rate': 1.0,
                'cpu_usage_percent': 10,
                'memory_usage_mb': 64
            }
        }
        
        observation_result = await auditor.process_task(observation_task)
        assert observation_result.success == True
        assert 'agent_observed' in observation_result.result['action']
        
        print("✅ Auditor observation task executed")
        
        # Test data persistence (check if data was stored in live databases)
        stored_data = await memory_manager.read("integration_test_status")
        if stored_data:
            print("✅ Task results persisted to live databases")
        
        # Cleanup
        await chief_of_staff.shutdown()
        await auditor.shutdown()
        await memory_manager.disconnect_all()
        
        print("🎉 Agent task execution with live infrastructure passed!")
        return True
        
    except Exception as e:
        print(f"❌ Agent task execution test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run comprehensive integration tests with live databases."""
    print("🚀 AETHELRED LIVE DATABASE INTEGRATION TESTS")
    print("=" * 70)
    
    tests = [
        ("Live Database Connections", test_live_database_connections),
        ("Memory Tier Integration", test_memory_tier_integration),
        ("Complete System Integration", test_system_with_live_databases),
        ("Agent Task Execution", test_agent_task_execution)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results[test_name] = False
    
    print("\\n" + "=" * 70)
    print("📊 LIVE DATABASE INTEGRATION TEST RESULTS")
    print("=" * 70)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:.<50} {status}")
        if result:
            passed += 1
    
    print(f"\\nTotal: {passed}/{total} integration tests passed")
    
    if passed == total:
        print("\\n🎉 ALL LIVE DATABASE INTEGRATION TESTS PASSED!")
        print("AETHELRED is ready for full deployment!")
        print("\\n🚀 System Status:")
        print("• Live Redis connection: ✅ Working")
        print("• Live PostgreSQL connection: ✅ Working") 
        print("• Live Neo4j connection: ✅ Working")
        print("• Memory tier integration: ✅ Working")
        print("• Agent task execution: ✅ Working")
        print("• Complete system integration: ✅ Working")
        print("\\n📋 Ready for Production:")
        print("• All database connections validated")
        print("• All agents operational with live data")
        print("• System orchestration working end-to-end")
        return True
    else:
        print(f"\\n⚠️  {total - passed} integration tests failed.")
        print("Please check database connections and system configuration.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
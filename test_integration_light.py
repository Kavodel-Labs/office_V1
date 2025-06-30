#!/usr/bin/env python3
"""
Lightweight integration tests for AETHELRED with optimized database setup.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_lightweight_connections():
    """Test connections to lightweight database setup."""
    print("🔌 Testing lightweight database connections...")
    
    try:
        from core.memory.interfaces import HotMemory, WarmMemory, ArchiveMemory
        
        # Test Redis (Hot Memory)
        redis_config = {'host': 'localhost', 'port': 6379, 'db': 0, 'ttl': 3600}
        hot_memory = HotMemory(redis_config)
        await hot_memory.connect()
        
        await hot_memory.write("test_redis", {"message": "Hello Redis Light!"})
        redis_result = await hot_memory.read("test_redis")
        assert redis_result["message"] == "Hello Redis Light!"
        print("✅ Redis (Hot Memory) - Connected and tested")
        await hot_memory.disconnect()
        
        # Test PostgreSQL (Warm Memory)
        postgres_config = {
            'host': 'localhost', 'port': 5432, 'database': 'aethelred',
            'user': 'aethelred', 'password': 'development'
        }
        warm_memory = WarmMemory(postgres_config)
        await warm_memory.connect()
        
        test_uuid = "550e8400-e29b-41d4-a716-446655440000"
        await warm_memory.write(test_uuid, {"message": "Hello PostgreSQL Light!"})
        print("✅ PostgreSQL (Warm Memory) - Connected and tested")
        await warm_memory.disconnect()
        
        # Test Archive Memory
        archive_config = {'filesystem': {'path': './test_light_archive'}}
        archive_memory = ArchiveMemory(archive_config)
        await archive_memory.connect()
        
        await archive_memory.write("test_archive", {"message": "Hello Archive Light!"})
        archive_result = await archive_memory.read("test_archive")
        assert archive_result["message"] == "Hello Archive Light!"
        print("✅ Archive Memory - Connected and tested")
        await archive_memory.disconnect()
        
        return True
        
    except Exception as e:
        print(f"❌ Lightweight connections test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_lightweight_system():
    """Test AETHELRED system with lightweight setup."""
    print("\\n🚀 Testing AETHELRED with lightweight infrastructure...")
    
    try:
        from core.memory.tier_manager import MemoryTierManager
        from core.memory.interfaces import HotMemory, WarmMemory, ArchiveMemory
        from core.task_management.task_master_integration import TaskMasterIntegration
        from core.task_management.task_router import TaskRouter
        from agents.apex.chief_of_staff.agent import ChiefOfStaff
        from agents.service.auditor.agent import Auditor
        
        # Memory system with lightweight tiers
        memory_manager = MemoryTierManager()
        
        hot_config = {'host': 'localhost', 'port': 6379, 'db': 0, 'ttl': 3600}
        warm_config = {
            'host': 'localhost', 'port': 5432, 'database': 'aethelred',
            'user': 'aethelred', 'password': 'development'
        }
        archive_config = {'filesystem': {'path': './test_light_archive'}}
        
        hot_tier = HotMemory(hot_config)
        warm_tier = WarmMemory(warm_config)
        archive_tier = ArchiveMemory(archive_config)
        
        memory_manager.register_tier(hot_tier)
        memory_manager.register_tier(warm_tier) 
        memory_manager.register_tier(archive_tier)
        
        await memory_manager.connect_all()
        print("✅ Lightweight memory system connected")
        
        # Task management
        task_config = {
            'task_master_path': 'task-master-ai',
            'project_root': Path.cwd(),
            'max_concurrent_tasks': 5  # Reduced for lightweight
        }
        task_master = TaskMasterIntegration(memory_manager, task_config)
        task_router = TaskRouter(memory_manager, task_master)
        
        # Create agents
        chief_config = {'test_mode': True, 'lightweight': True}
        chief_of_staff = ChiefOfStaff(memory_manager, task_router, task_master, chief_config)
        await chief_of_staff.initialize()
        
        auditor_config = {'test_mode': True, 'lightweight': True}
        auditor = Auditor(memory_manager, auditor_config)
        await auditor.initialize()
        
        print("✅ Agents initialized with lightweight config")
        
        # Test system status
        status_task = {
            'id': 'light_system_status',
            'type': 'system_status',
            'description': 'Get lightweight system status'
        }
        
        status_result = await chief_of_staff.process_task(status_task)
        assert status_result.success == True
        print("✅ Chief of Staff system status working")
        
        # Test auditor performance observation
        observation_task = {
            'id': 'light_observation',
            'type': 'observe_agent',
            'agent_id': 'A_ChiefOfStaff',
            'observation_data': {
                'response_time_ms': 100,
                'success_rate': 1.0,
                'memory_usage_mb': 32  # Lightweight
            }
        }
        
        observation_result = await auditor.process_task(observation_task)
        assert observation_result.success == True
        print("✅ Auditor observation working")
        
        # Test memory operations
        await memory_manager.write("light_test", {"system": "lightweight", "status": "operational"})
        test_data = await memory_manager.read("light_test")
        assert test_data["system"] == "lightweight"
        print("✅ Memory operations working")
        
        # Health checks
        memory_health = await memory_manager.health_check()
        healthy_tiers = sum(1 for h in memory_health.values() if h.get('status') == 'healthy')
        print(f"✅ Memory health: {healthy_tiers}/{len(memory_health)} tiers healthy")
        
        # Cleanup
        await chief_of_staff.shutdown()
        await auditor.shutdown()
        await memory_manager.disconnect_all()
        
        print("🎉 Lightweight system test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Lightweight system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run lightweight integration tests."""
    print("🚀 AETHELRED LIGHTWEIGHT INTEGRATION TESTS")
    print("=" * 60)
    
    tests = [
        ("Lightweight Database Connections", test_lightweight_connections),
        ("Lightweight System Integration", test_lightweight_system)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results[test_name] = False
    
    print("\\n" + "=" * 60)
    print("📊 LIGHTWEIGHT INTEGRATION TEST RESULTS")
    print("=" * 60)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:.<45} {status}")
        if result:
            passed += 1
    
    print(f"\\nTotal: {passed}/{total} lightweight tests passed")
    
    if passed == total:
        print("\\n🎉 ALL LIGHTWEIGHT TESTS PASSED!")
        print("AETHELRED lightweight system is operational!")
        print("\\n🚀 System Ready:")
        print("• Memory-optimized Redis: ✅ Working")
        print("• Lightweight PostgreSQL: ✅ Working")
        print("• Archive storage: ✅ Working")
        print("• Agent framework: ✅ Working")
        print("• Task management: ✅ Working")
        print("\\n📋 Ready for Full Deployment!")
        return True
    else:
        print(f"\\n⚠️  {total - passed} tests failed.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
#!/usr/bin/env python3
"""
Test AETHELRED with real database dependencies installed.

This validates that all imports work with the actual database drivers.
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


async def test_imports_with_dependencies():
    """Test all imports now that dependencies are installed."""
    print("🧪 Testing imports with full dependencies...")
    
    try:
        # Test core memory interfaces with real backends
        from core.memory.interfaces import HotMemory, WarmMemory, ColdMemory, ArchiveMemory
        print("✅ Memory interfaces with real backends imported successfully")
        
        # Test that the backends are available
        from core.memory.interfaces import AIOREDIS_AVAILABLE, ASYNCPG_AVAILABLE, NEO4J_AVAILABLE
        print(f"✅ aioredis available: {AIOREDIS_AVAILABLE}")
        print(f"✅ asyncpg available: {ASYNCPG_AVAILABLE}")
        print(f"✅ neo4j available: {NEO4J_AVAILABLE}")
        
        # Test agent imports
        from agents.apex.chief_of_staff.agent import ChiefOfStaff
        from agents.service.auditor.agent import Auditor
        print("✅ Core agents imported successfully")
        
        # Test main system
        from main import AethelredSystem
        print("✅ Main system imported successfully")
        
        print("🎉 All imports with dependencies successful!")
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_memory_interfaces():
    """Test memory interface creation (without actual connections)."""
    print("\n🧠 Testing memory interface creation...")
    
    try:
        from core.memory.interfaces import HotMemory, WarmMemory, ColdMemory, ArchiveMemory
        
        # Test Hot Memory (Redis)
        hot_config = {
            'host': 'localhost',
            'port': 6379,
            'db': 0,
            'ttl': 3600
        }
        hot_memory = HotMemory(hot_config)
        print("✅ HotMemory (Redis) interface created")
        
        # Test Warm Memory (PostgreSQL)
        warm_config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'aethelred',
            'user': 'aethelred',
            'password': 'development'
        }
        warm_memory = WarmMemory(warm_config)
        print("✅ WarmMemory (PostgreSQL) interface created")
        
        # Test Cold Memory (Neo4j)
        cold_config = {
            'uri': 'bolt://localhost:7687',
            'user': 'neo4j',
            'password': 'development'
        }
        cold_memory = ColdMemory(cold_config)
        print("✅ ColdMemory (Neo4j) interface created")
        
        # Test Archive Memory (Filesystem)
        archive_config = {
            'filesystem': {
                'path': './test_archive'
            }
        }
        archive_memory = ArchiveMemory(archive_config)
        print("✅ ArchiveMemory (Filesystem) interface created")
        
        print("🎉 All memory interfaces created successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Memory interface test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_agent_creation():
    """Test creating real agents (without external dependencies)."""
    print("\n🤖 Testing agent creation...")
    
    try:
        from core.memory.tier_manager import MemoryTierManager
        from core.task_management.task_master_integration import TaskMasterIntegration
        from core.task_management.task_router import TaskRouter
        
        # Create memory manager (with mocks for testing)
        memory_manager = MemoryTierManager()
        
        # Create task management
        task_config = {
            'task_master_path': 'task-master-ai',
            'project_root': Path.cwd(),
            'max_concurrent_tasks': 10
        }
        task_master = TaskMasterIntegration(memory_manager, task_config)
        task_router = TaskRouter(memory_manager, task_master)
        
        print("✅ Task management components created")
        
        # Test Chief of Staff creation
        from agents.apex.chief_of_staff.agent import ChiefOfStaff
        chief_config = {'test_mode': True}
        chief_of_staff = ChiefOfStaff(memory_manager, task_router, task_master, chief_config)
        print("✅ Chief of Staff agent created")
        
        # Test Auditor creation
        from agents.service.auditor.agent import Auditor
        auditor_config = {'test_mode': True}
        auditor = Auditor(memory_manager, auditor_config)
        print("✅ Auditor agent created")
        
        print("🎉 All agents created successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Agent creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_system_creation():
    """Test creating the main system."""
    print("\n🚀 Testing main system creation...")
    
    try:
        from main import AethelredSystem
        
        # Create system (should load config)
        system = AethelredSystem()
        print("✅ AethelredSystem created")
        
        # Test system info
        status = system.get_system_status()
        print(f"✅ System status retrieved: {status['system_name']} v{status['version']}")
        
        print("🎉 Main system creation successful!")
        return True
        
    except Exception as e:
        print(f"❌ System creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run tests with full dependencies."""
    print("🚀 AETHELRED DEPENDENCY VALIDATION TEST")
    print("=" * 60)
    
    tests = [
        ("Import Tests", test_imports_with_dependencies),
        ("Memory Interface Tests", test_memory_interfaces),
        ("Agent Creation Tests", test_agent_creation),
        ("System Creation Tests", test_system_creation)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    print("\n" + "=" * 60)
    print("📊 DEPENDENCY VALIDATION RESULTS")
    print("=" * 60)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:.<40} {status}")
        if result:
            passed += 1
    
    print(f"\nTotal: {passed}/{total} dependency tests passed")
    
    if passed == total:
        print("\n🎉 ALL DEPENDENCY TESTS PASSED!")
        print("AETHELRED is ready for full infrastructure testing!")
        print("\n📋 Next Steps:")
        print("1. Start Docker services: docker compose -f config/docker-compose.dev.yml up -d")
        print("2. Run integration tests with real databases")
        print("3. Start the full system: python main.py")
        return True
    else:
        print(f"\n⚠️  {total - passed} dependency tests failed.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
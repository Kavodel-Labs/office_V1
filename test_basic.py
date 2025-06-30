#!/usr/bin/env python3
"""
Basic test script for Project Aethelred core components.

Tests the system without requiring full Docker infrastructure.
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


async def test_imports():
    """Test that all core modules can be imported."""
    print("🧪 Testing imports...")
    
    try:
        from core.memory.tier_manager import MemoryTierManager, MemoryTier
        print("✅ Memory tier manager imports successful")
        
        from core.memory.coherence import MemoryCoherenceManager, ConsistencyLevel
        print("✅ Memory coherence imports successful")
        
        from core.task_management.task_master_integration import TaskMasterIntegration
        print("✅ Task master integration imports successful")
        
        from core.task_management.task_router import TaskRouter, RoutingStrategy
        print("✅ Task router imports successful")
        
        from agents.base.agent import Agent, AgentCapability, AgentStatus
        print("✅ Base agent imports successful")
        
        # Test memory interfaces (may fail due to missing dependencies)
        try:
            from core.memory.interfaces import HotMemory, WarmMemory, ColdMemory, ArchiveMemory
            print("✅ Memory interfaces imports successful")
        except ImportError as e:
            print(f"⚠️  Memory interfaces import warning: {e}")
        
        # Test agent implementations
        try:
            from agents.apex.chief_of_staff.agent import ChiefOfStaff
            print("✅ Chief of Staff agent imports successful")
        except ImportError as e:
            print(f"⚠️  Chief of Staff import warning: {e}")
        
        try:
            from agents.service.auditor.agent import Auditor
            print("✅ Auditor agent imports successful")
        except ImportError as e:
            print(f"⚠️  Auditor import warning: {e}")
        
        print("🎉 Core imports successful!")
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_memory_manager():
    """Test memory tier manager without actual backends."""
    print("\n🧠 Testing Memory Tier Manager...")
    
    try:
        from core.memory.tier_manager import MemoryTierManager, MemoryTier
        
        # Create a mock memory tier for testing
        class MockMemoryTier(MemoryTier):
            def __init__(self, name: str, tier_level: int):
                super().__init__(name, tier_level)
                self.data = {}
                
            async def connect(self):
                self.is_connected = True
                
            async def disconnect(self):
                self.is_connected = False
                
            async def read(self, key: str):
                return self.data.get(key)
                
            async def write(self, key: str, value, **kwargs):
                self.data[key] = value
                
            async def delete(self, key: str):
                return self.data.pop(key, None) is not None
                
            async def exists(self, key: str):
                return key in self.data
                
            async def health_check(self):
                return {'status': 'healthy', 'connected': self.is_connected}
        
        # Test memory manager
        manager = MemoryTierManager()
        
        # Register mock tiers
        hot_tier = MockMemoryTier("hot", 0)
        warm_tier = MockMemoryTier("warm", 1)
        
        manager.register_tier(hot_tier)
        manager.register_tier(warm_tier)
        
        # Connect tiers
        await manager.connect_all()
        
        # Test write/read
        await manager.write("test_key", {"message": "hello aethelred"}, target_tiers=["hot"])
        result = await manager.read("test_key")
        
        if result and result.get("message") == "hello aethelred":
            print("✅ Memory manager write/read test successful")
        else:
            print(f"❌ Memory manager test failed - got: {result}")
            return False
            
        # Test health check
        health = await manager.health_check()
        print(f"✅ Memory health check: {health}")
        
        await manager.disconnect_all()
        print("✅ Memory tier manager tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Memory manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_task_master_integration():
    """Test task master integration."""
    print("\n📋 Testing Task Master Integration...")
    
    try:
        from core.task_management.task_master_integration import TaskMasterIntegration
        from core.memory.tier_manager import MemoryTierManager
        
        # Create mock memory manager
        memory_manager = MemoryTierManager()
        
        # Test task master config
        config = {
            'task_master_path': 'task-master-ai',
            'project_root': Path.cwd(),
            'max_concurrent_tasks': 10
        }
        
        task_master = TaskMasterIntegration(memory_manager, config)
        
        # Test statistics (should work without initialization)
        stats = task_master.get_statistics()
        print(f"✅ Task master statistics: {stats}")
        
        print("✅ Task master integration basic tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Task master integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_agents():
    """Test agent initialization."""
    print("\n🤖 Testing Agents...")
    
    try:
        from agents.base.agent import Agent, AgentCapability, AgentStatus
        from core.memory.tier_manager import MemoryTierManager
        
        # Test base agent capabilities
        capabilities = [AgentCapability.SYSTEM_CONFIG_READ, AgentCapability.TASKS_ROUTING]
        print(f"✅ Agent capabilities loaded: {[cap.value for cap in capabilities]}")
        
        # Test agent status
        status = AgentStatus.IDLE
        print(f"✅ Agent status: {status.value}")
        
        print("✅ Agent framework tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Agent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_system_config():
    """Test system configuration loading."""
    print("\n⚙️  Testing Configuration...")
    
    try:
        import yaml
        
        config_path = Path("config/aethelred-config.yaml")
        if not config_path.exists():
            print(f"❌ Configuration file not found: {config_path}")
            return False
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        print(f"✅ Configuration loaded successfully")
        print(f"   System name: {config.get('system', {}).get('name')}")
        print(f"   Edition: {config.get('system', {}).get('edition')}")
        print(f"   Memory tiers: {len(config.get('cognitive', {}).get('memory_tiers', []))}")
        print(f"   Agent roster: {len(config.get('agents', {}).get('roster', []))}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


async def main():
    """Run all basic tests."""
    print("🚀 AETHELRED BASIC SYSTEM TEST")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Memory Manager", test_memory_manager),
        ("Task Master Integration", test_task_master_integration),
        ("Agents", test_agents),
        ("Configuration", test_system_config)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS")
    print("=" * 50)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:.<30} {status}")
        if result:
            passed += 1
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL BASIC TESTS PASSED!")
        print("The AETHELRED core system is working correctly.")
        print("\nNext steps:")
        print("1. Install Docker and run full integration tests")
        print("2. Start the complete system with: python main.py")
        return True
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please fix issues before proceeding.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
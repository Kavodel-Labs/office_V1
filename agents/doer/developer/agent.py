"""
Developer Agent for Project Aethelred.

The Developer is a Doer tier agent specialized in:
- MCP (Model Context Protocol) server creation and debugging
- Integration development and testing
- Code generation for protocol implementations
- Debugging complex integration issues
- Creating development tools and utilities
"""

import asyncio
import logging
import json
import subprocess
import os
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path

from agents.base.agent import Agent, AgentCapability, AgentStatus, TaskResult
from core.memory.tier_manager import MemoryTierManager

logger = logging.getLogger(__name__)


@dataclass
class MCPServerSpec:
    """Specification for an MCP server."""
    name: str
    description: str
    protocol_version: str
    capabilities: List[str]
    endpoints: List[Dict[str, Any]]
    dependencies: List[str]
    config_schema: Dict[str, Any]


@dataclass
class IntegrationTest:
    """Integration test specification."""
    name: str
    test_type: str  # unit, integration, e2e
    target: str  # what's being tested
    steps: List[str]
    expected_results: List[str]
    actual_results: Optional[List[str]] = None
    status: str = "pending"  # pending, running, passed, failed


class Developer(Agent):
    """
    Developer - MCP Server Creation and Debugging Specialist.
    
    Responsibilities:
    - Create and maintain MCP servers
    - Debug protocol integration issues
    - Generate integration code and tests
    - Optimize communication protocols
    - Develop development tools and utilities
    - Troubleshoot complex technical issues
    """
    
    def __init__(self, memory_manager: MemoryTierManager,
                 config: Optional[Dict[str, Any]] = None):
        
        super().__init__(
            agent_id="D_Developer",
            version=1,
            tier="doer",
            role="MCP Development Specialist",
            capabilities=[
                AgentCapability.CODE_BACKEND_DEVELOP,
                AgentCapability.CODE_FRONTEND_DEVELOP,
                AgentCapability.CODE_REVIEW,
                AgentCapability.TESTS_WRITE,
                AgentCapability.TESTS_EXECUTE,
                AgentCapability.TASKS_EXECUTE
            ],
            config=config or {}
        )
        
        self.memory_manager = memory_manager
        
        # Development state
        self.active_projects: Dict[str, Dict[str, Any]] = {}
        self.mcp_servers: Dict[str, MCPServerSpec] = {}
        self.integration_tests: List[IntegrationTest] = []
        self.debug_sessions: Dict[str, Dict[str, Any]] = {}
        
        # Development tools
        self.supported_languages = ['javascript', 'typescript', 'python', 'go', 'rust']
        self.mcp_frameworks = ['@modelcontextprotocol/sdk', 'mcp-python', 'mcp-go']
        self.testing_frameworks = ['jest', 'mocha', 'pytest', 'go test']
        
        # Statistics
        self.servers_created = 0
        self.bugs_fixed = 0
        self.tests_written = 0
        self.integrations_completed = 0
        
    async def on_initialize(self) -> None:
        """Initialize Developer specific resources."""
        logger.info("Initializing Developer agent...")
        
        # Load existing projects
        await self._load_project_history()
        
        # Set up development environment
        await self._setup_dev_environment()
        
        # Load MCP templates and examples
        await self._load_mcp_templates()
        
        logger.info("Developer initialization complete")
        
    async def execute_task(self, task: Dict[str, Any]) -> Any:
        """
        Execute a development task.
        
        Args:
            task: Task to execute
            
        Returns:
            Task execution result
        """
        task_type = task.get('type')
        
        if task_type == 'create_mcp_server':
            return await self._handle_create_mcp_server(task)
        elif task_type == 'debug_mcp_integration':
            return await self._handle_debug_mcp_integration(task)
        elif task_type == 'write_integration_tests':
            return await self._handle_write_integration_tests(task)
        elif task_type == 'review_mcp_code':
            return await self._handle_review_mcp_code(task)
        elif task_type == 'optimize_protocol':
            return await self._handle_optimize_protocol(task)
        elif task_type == 'create_dev_tools':
            return await self._handle_create_dev_tools(task)
        elif task_type == 'troubleshoot_issue':
            return await self._handle_troubleshoot_issue(task)
        elif task_type == 'generate_documentation':
            return await self._handle_generate_documentation(task)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
            
    async def validate_task(self, task: Dict[str, Any]) -> bool:
        """
        Validate if the Developer can execute the task.
        
        Args:
            task: Task to validate
            
        Returns:
            True if task can be executed
        """
        valid_task_types = {
            'create_mcp_server',
            'debug_mcp_integration',
            'write_integration_tests',
            'review_mcp_code',
            'optimize_protocol',
            'create_dev_tools',
            'troubleshoot_issue',
            'generate_documentation'
        }
        
        task_type = task.get('type')
        return task_type in valid_task_types
        
    async def check_agent_health(self) -> Dict[str, Any]:
        """Developer specific health checks."""
        health_data = {
            'servers_created': self.servers_created,
            'bugs_fixed': self.bugs_fixed,
            'tests_written': self.tests_written,
            'integrations_completed': self.integrations_completed,
            'active_projects': len(self.active_projects),
            'debug_sessions': len(self.debug_sessions),
            'dev_environment_status': 'healthy'
        }
        
        # Check development tools availability
        try:
            dev_tools_status = await self._check_dev_tools()
            health_data['dev_tools'] = dev_tools_status
        except Exception as e:
            health_data['dev_tools_error'] = str(e)
            
        return health_data
        
    # Task handlers
    
    async def _handle_create_mcp_server(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new MCP server."""
        server_spec = task.get('server_spec', {})
        language = task.get('language', 'typescript')
        framework = task.get('framework', '@modelcontextprotocol/sdk')
        
        logger.info(f"Creating MCP server: {server_spec.get('name', 'unnamed')}")
        
        try:
            # Generate server specification
            spec = MCPServerSpec(
                name=server_spec.get('name', 'new-mcp-server'),
                description=server_spec.get('description', 'New MCP server'),
                protocol_version=server_spec.get('protocol_version', '1.0.0'),
                capabilities=server_spec.get('capabilities', ['chat', 'tools']),
                endpoints=server_spec.get('endpoints', []),
                dependencies=server_spec.get('dependencies', []),
                config_schema=server_spec.get('config_schema', {})
            )
            
            # Create server code
            server_code = await self._generate_mcp_server_code(spec, language, framework)
            
            # Create project structure
            project_path = await self._create_project_structure(spec, language)
            
            # Write server files
            files_created = await self._write_server_files(project_path, server_code, spec)
            
            # Generate tests
            test_files = await self._generate_server_tests(spec, language)
            
            # Write documentation
            docs = await self._generate_server_documentation(spec)
            
            self.servers_created += 1
            self.mcp_servers[spec.name] = spec
            
            # Store project info
            project_info = {
                'spec': spec,
                'language': language,
                'framework': framework,
                'project_path': str(project_path),
                'files_created': files_created,
                'created_at': datetime.utcnow().isoformat(),
                'status': 'created'
            }
            self.active_projects[spec.name] = project_info
            
            return {
                'action': 'mcp_server_created',
                'server_name': spec.name,
                'language': language,
                'framework': framework,
                'project_path': str(project_path),
                'files_created': len(files_created),
                'test_files': len(test_files),
                'capabilities': spec.capabilities,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to create MCP server: {e}")
            return {
                'action': 'mcp_server_creation_failed',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
            
    async def _handle_debug_mcp_integration(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Debug MCP integration issues."""
        integration_name = task.get('integration_name', 'unknown')
        issue_description = task.get('issue_description', '')
        logs = task.get('logs', [])
        
        logger.info(f"Debugging MCP integration: {integration_name}")
        
        try:
            # Analyze the issue
            issue_analysis = await self._analyze_integration_issue(
                integration_name, issue_description, logs
            )
            
            # Generate debug steps
            debug_steps = await self._generate_debug_steps(issue_analysis)
            
            # Execute debugging
            debug_results = await self._execute_debug_steps(debug_steps)
            
            # Generate fix recommendations
            fix_recommendations = await self._generate_fix_recommendations(
                issue_analysis, debug_results
            )
            
            # Create debug session
            session_id = f"debug_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            debug_session = {
                'session_id': session_id,
                'integration_name': integration_name,
                'issue_description': issue_description,
                'issue_analysis': issue_analysis,
                'debug_steps': debug_steps,
                'debug_results': debug_results,
                'fix_recommendations': fix_recommendations,
                'created_at': datetime.utcnow().isoformat(),
                'status': 'completed'
            }
            
            self.debug_sessions[session_id] = debug_session
            self.bugs_fixed += 1
            
            return {
                'action': 'mcp_integration_debugged',
                'session_id': session_id,
                'integration_name': integration_name,
                'issue_analysis': issue_analysis,
                'fix_recommendations': fix_recommendations,
                'debug_steps_executed': len(debug_steps),
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to debug MCP integration: {e}")
            return {
                'action': 'mcp_debug_failed',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
            
    async def _handle_write_integration_tests(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Write integration tests for MCP components."""
        component = task.get('component', '')
        test_scenarios = task.get('test_scenarios', [])
        test_framework = task.get('test_framework', 'jest')
        
        logger.info(f"Writing integration tests for: {component}")
        
        try:
            # Generate test specifications
            test_specs = []
            for scenario in test_scenarios:
                test_spec = IntegrationTest(
                    name=scenario.get('name', 'unnamed_test'),
                    test_type=scenario.get('type', 'integration'),
                    target=component,
                    steps=scenario.get('steps', []),
                    expected_results=scenario.get('expected_results', [])
                )
                test_specs.append(test_spec)
                
            # Generate test code
            test_files = await self._generate_test_code(test_specs, test_framework)
            
            # Write test files
            tests_written = await self._write_test_files(test_files, component)
            
            # Generate test configuration
            test_config = await self._generate_test_config(component, test_framework)
            
            self.tests_written += len(test_specs)
            self.integration_tests.extend(test_specs)
            
            return {
                'action': 'integration_tests_written',
                'component': component,
                'test_framework': test_framework,
                'tests_created': len(test_specs),
                'test_files': list(test_files.keys()),
                'config_generated': test_config is not None,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to write integration tests: {e}")
            return {
                'action': 'test_writing_failed',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
            
    async def _handle_troubleshoot_issue(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Troubleshoot complex technical issues."""
        issue_type = task.get('issue_type', 'general')
        description = task.get('description', '')
        context = task.get('context', {})
        
        logger.info(f"Troubleshooting {issue_type} issue")
        
        try:
            # Analyze the issue
            analysis = await self._analyze_technical_issue(issue_type, description, context)
            
            # Generate investigation plan
            investigation_plan = await self._create_investigation_plan(analysis)
            
            # Execute investigation
            investigation_results = await self._execute_investigation(investigation_plan)
            
            # Generate solution recommendations
            solutions = await self._generate_solution_recommendations(
                analysis, investigation_results
            )
            
            return {
                'action': 'issue_troubleshooted',
                'issue_type': issue_type,
                'analysis': analysis,
                'investigation_steps': len(investigation_plan),
                'solutions_found': len(solutions),
                'recommendations': solutions,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to troubleshoot issue: {e}")
            return {
                'action': 'troubleshooting_failed',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    # Core development methods
    
    async def _generate_mcp_server_code(self, spec: MCPServerSpec, language: str, framework: str) -> Dict[str, str]:
        """Generate MCP server code based on specification."""
        code_files = {}
        
        if language == 'typescript':
            # Generate TypeScript MCP server
            code_files['src/server.ts'] = self._generate_typescript_server(spec, framework)
            code_files['src/handlers.ts'] = self._generate_typescript_handlers(spec)
            code_files['src/types.ts'] = self._generate_typescript_types(spec)
            code_files['package.json'] = self._generate_package_json(spec, framework)
            code_files['tsconfig.json'] = self._generate_tsconfig()
            
        elif language == 'python':
            # Generate Python MCP server
            code_files['src/server.py'] = self._generate_python_server(spec, framework)
            code_files['src/handlers.py'] = self._generate_python_handlers(spec)
            code_files['src/types.py'] = self._generate_python_types(spec)
            code_files['requirements.txt'] = self._generate_requirements(spec, framework)
            code_files['setup.py'] = self._generate_setup_py(spec)
            
        return code_files
        
    async def _create_project_structure(self, spec: MCPServerSpec, language: str) -> Path:
        """Create project directory structure."""
        project_name = spec.name.replace(' ', '-').lower()
        project_path = Path(f"./mcp_servers/{project_name}")
        
        # Create directories
        directories = ['src', 'tests', 'docs', 'examples']
        for directory in directories:
            (project_path / directory).mkdir(parents=True, exist_ok=True)
            
        return project_path
        
    async def _write_server_files(self, project_path: Path, code_files: Dict[str, str], spec: MCPServerSpec) -> List[str]:
        """Write server files to disk."""
        files_written = []
        
        for file_path, content in code_files.items():
            full_path = project_path / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(full_path, 'w') as f:
                f.write(content)
                
            files_written.append(str(full_path))
            
        return files_written
        
    def _generate_typescript_server(self, spec: MCPServerSpec, framework: str) -> str:
        """Generate TypeScript MCP server code."""
        return f'''/**
 * {spec.name} - MCP Server
 * {spec.description}
 * 
 * Generated by AETHELRED Developer Agent
 */

import {{ Server }} from '{framework}';
import {{ StdioServerTransport }} from '{framework}/stdio';
import * as handlers from './handlers';
import {{ {spec.name}Config }} from './types';

class {spec.name.replace('-', '')}Server extends Server {{
    private config: {spec.name}Config;
    
    constructor(config: {spec.name}Config) {{
        super({{
            name: '{spec.name}',
            version: '{spec.protocol_version}',
            capabilities: {json.dumps(spec.capabilities, indent=8)}
        }});
        
        this.config = config;
        this.setupHandlers();
    }}
    
    private setupHandlers(): void {{
        // Setup MCP handlers
        {self._generate_handler_setup(spec)}
    }}
}}

async function main() {{
    const config: {spec.name}Config = {{
        // Add configuration here
    }};
    
    const server = new {spec.name.replace('-', '')}Server(config);
    const transport = new StdioServerTransport();
    
    await server.connect(transport);
    console.log(`{spec.name} MCP Server running on stdio`);
}}

if (require.main === module) {{
    main().catch(console.error);
}}

export {{ {spec.name.replace('-', '')}Server }};
'''
        
    def _generate_typescript_handlers(self, spec: MCPServerSpec) -> str:
        """Generate TypeScript handler functions."""
        handlers = []
        
        for endpoint in spec.endpoints:
            handler_name = endpoint.get('name', 'handler')
            handler_code = f'''
export async function {handler_name}(params: any): Promise<any> {{
    // Implementation for {handler_name}
    return {{
        success: true,
        data: params
    }};
}}'''
            handlers.append(handler_code)
            
        return '\n'.join(handlers)
        
    def _generate_python_server(self, spec: MCPServerSpec, framework: str) -> str:
        """Generate Python MCP server code."""
        return f'''"""
{spec.name} - MCP Server
{spec.description}

Generated by AETHELRED Developer Agent
"""

import asyncio
import logging
from typing import Any, Dict, List
from {framework} import Server, StdioServerTransport
from .handlers import *
from .types import {spec.name.replace('-', '')}Config

logger = logging.getLogger(__name__)


class {spec.name.replace('-', '')}Server(Server):
    def __init__(self, config: {spec.name.replace('-', '')}Config):
        super().__init__(
            name="{spec.name}",
            version="{spec.protocol_version}",
            capabilities={spec.capabilities}
        )
        
        self.config = config
        self.setup_handlers()
        
    def setup_handlers(self):
        """Setup MCP handlers."""
        {self._generate_python_handler_setup(spec)}


async def main():
    config = {spec.name.replace('-', '')}Config()
    
    server = {spec.name.replace('-', '')}Server(config)
    transport = StdioServerTransport()
    
    await server.connect(transport)
    logger.info(f"{spec.name} MCP Server running on stdio")


if __name__ == "__main__":
    asyncio.run(main())
'''
        
    def _generate_handler_setup(self, spec: MCPServerSpec) -> str:
        """Generate handler setup code."""
        setup_lines = []
        for endpoint in spec.endpoints:
            name = endpoint.get('name', 'handler')
            setup_lines.append(f"        this.setHandler('{name}', handlers.{name});")
        return '\n'.join(setup_lines)
        
    def _generate_python_handler_setup(self, spec: MCPServerSpec) -> str:
        """Generate Python handler setup code."""
        setup_lines = []
        for endpoint in spec.endpoints:
            name = endpoint.get('name', 'handler')
            setup_lines.append(f"        self.set_handler('{name}', {name})")
        return '\n'.join(setup_lines)
        
    async def _analyze_integration_issue(self, integration_name: str, description: str, logs: List[str]) -> Dict[str, Any]:
        """Analyze MCP integration issues."""
        analysis = {
            'issue_category': 'unknown',
            'severity': 'medium',
            'likely_causes': [],
            'affected_components': [],
            'log_patterns': []
        }
        
        # Analyze logs for common patterns
        error_patterns = {
            'connection_refused': ['ECONNREFUSED', 'connection refused'],
            'timeout': ['timeout', 'ETIMEDOUT'],
            'auth_failure': ['unauthorized', 'authentication failed'],
            'protocol_mismatch': ['protocol version', 'unsupported version'],
            'missing_capability': ['capability not found', 'unsupported capability']
        }
        
        for pattern_name, patterns in error_patterns.items():
            for log_line in logs:
                if any(pattern.lower() in log_line.lower() for pattern in patterns):
                    analysis['log_patterns'].append(pattern_name)
                    analysis['likely_causes'].append(pattern_name)
                    
        # Categorize based on description
        if 'connection' in description.lower():
            analysis['issue_category'] = 'connectivity'
        elif 'authentication' in description.lower() or 'auth' in description.lower():
            analysis['issue_category'] = 'authentication'
        elif 'protocol' in description.lower():
            analysis['issue_category'] = 'protocol'
        elif 'performance' in description.lower() or 'slow' in description.lower():
            analysis['issue_category'] = 'performance'
            
        return analysis
        
    async def _generate_debug_steps(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate debugging steps based on analysis."""
        steps = ['Check system logs for additional error details']
        
        issue_category = analysis.get('issue_category', 'unknown')
        
        if issue_category == 'connectivity':
            steps.extend([
                'Test network connectivity to MCP server',
                'Verify MCP server is running and listening',
                'Check firewall and port configurations',
                'Validate server address and port settings'
            ])
        elif issue_category == 'authentication':
            steps.extend([
                'Verify authentication tokens and credentials',
                'Check token expiration dates',
                'Validate permission scopes',
                'Test authentication with minimal permissions'
            ])
        elif issue_category == 'protocol':
            steps.extend([
                'Check MCP protocol version compatibility',
                'Validate message format and structure',
                'Test with protocol debugging enabled',
                'Compare with working protocol examples'
            ])
            
        return steps
        
    async def _check_dev_tools(self) -> Dict[str, str]:
        """Check availability of development tools."""
        tools_status = {}
        
        # Check Node.js
        try:
            result = subprocess.run(['node', '--version'], capture_output=True, text=True)
            tools_status['nodejs'] = result.stdout.strip() if result.returncode == 0 else 'not_available'
        except FileNotFoundError:
            tools_status['nodejs'] = 'not_installed'
            
        # Check Python
        try:
            result = subprocess.run(['python3', '--version'], capture_output=True, text=True)
            tools_status['python'] = result.stdout.strip() if result.returncode == 0 else 'not_available'
        except FileNotFoundError:
            tools_status['python'] = 'not_installed'
            
        # Check npm
        try:
            result = subprocess.run(['npm', '--version'], capture_output=True, text=True)
            tools_status['npm'] = result.stdout.strip() if result.returncode == 0 else 'not_available'
        except FileNotFoundError:
            tools_status['npm'] = 'not_installed'
            
        return tools_status
        
    # Placeholder methods for additional functionality
    async def _load_project_history(self) -> None:
        """Load existing project history."""
        pass
        
    async def _setup_dev_environment(self) -> None:
        """Set up development environment."""
        pass
        
    async def _load_mcp_templates(self) -> None:
        """Load MCP templates and examples."""
        pass
        
    async def _generate_server_tests(self, spec: MCPServerSpec, language: str) -> Dict[str, str]:
        """Generate test files for the server."""
        return {}
        
    async def _generate_server_documentation(self, spec: MCPServerSpec) -> str:
        """Generate documentation for the server."""
        return f"# {spec.name}\n\n{spec.description}"
        
    async def _execute_debug_steps(self, steps: List[str]) -> List[Dict[str, Any]]:
        """Execute debugging steps."""
        return [{'step': step, 'result': 'executed'} for step in steps]
        
    async def _generate_fix_recommendations(self, analysis: Dict[str, Any], debug_results: List[Dict[str, Any]]) -> List[str]:
        """Generate fix recommendations."""
        return ['Review configuration settings', 'Update dependencies']
        
    async def _generate_test_code(self, test_specs: List[IntegrationTest], framework: str) -> Dict[str, str]:
        """Generate test code files."""
        return {}
        
    async def _write_test_files(self, test_files: Dict[str, str], component: str) -> int:
        """Write test files to disk."""
        return len(test_files)
        
    async def _generate_test_config(self, component: str, framework: str) -> Optional[Dict[str, Any]]:
        """Generate test configuration."""
        return {}
        
    async def _analyze_technical_issue(self, issue_type: str, description: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze technical issues."""
        return {'category': issue_type, 'severity': 'medium'}
        
    async def _create_investigation_plan(self, analysis: Dict[str, Any]) -> List[str]:
        """Create investigation plan."""
        return ['Gather more information', 'Test hypothesis']
        
    async def _execute_investigation(self, plan: List[str]) -> List[Dict[str, Any]]:
        """Execute investigation steps."""
        return [{'step': step, 'result': 'completed'} for step in plan]
        
    async def _generate_solution_recommendations(self, analysis: Dict[str, Any], results: List[Dict[str, Any]]) -> List[str]:
        """Generate solution recommendations."""
        return ['Apply recommended fix', 'Monitor for resolution']
        
    def _generate_typescript_types(self, spec: MCPServerSpec) -> str:
        """Generate TypeScript type definitions."""
        return f"""export interface {spec.name}Config {{
    // Configuration interface
}}"""
        
    def _generate_python_types(self, spec: MCPServerSpec) -> str:
        """Generate Python type definitions."""
        return f"""from dataclasses import dataclass

@dataclass
class {spec.name.replace('-', '')}Config:
    pass"""
        
    def _generate_package_json(self, spec: MCPServerSpec, framework: str) -> str:
        """Generate package.json for TypeScript project."""
        return json.dumps({
            "name": spec.name,
            "version": spec.protocol_version,
            "description": spec.description,
            "main": "dist/server.js",
            "scripts": {
                "build": "tsc",
                "start": "node dist/server.js",
                "dev": "ts-node src/server.ts",
                "test": "jest"
            },
            "dependencies": {
                framework: "latest"
            },
            "devDependencies": {
                "typescript": "^5.0.0",
                "ts-node": "^10.0.0",
                "@types/node": "^20.0.0",
                "jest": "^29.0.0"
            }
        }, indent=2)
        
    def _generate_tsconfig(self) -> str:
        """Generate TypeScript configuration."""
        return json.dumps({
            "compilerOptions": {
                "target": "ES2020",
                "module": "commonjs",
                "lib": ["ES2020"],
                "outDir": "./dist",
                "rootDir": "./src",
                "strict": True,
                "esModuleInterop": True,
                "skipLibCheck": True,
                "forceConsistentCasingInFileNames": True
            },
            "include": ["src/**/*"],
            "exclude": ["node_modules", "dist"]
        }, indent=2)
        
    def _generate_requirements(self, spec: MCPServerSpec, framework: str) -> str:
        """Generate Python requirements.txt."""
        return f"""{framework}
asyncio
logging
typing"""
        
    def _generate_setup_py(self, spec: MCPServerSpec) -> str:
        """Generate Python setup.py."""
        return f"""from setuptools import setup, find_packages

setup(
    name="{spec.name}",
    version="{spec.protocol_version}",
    description="{spec.description}",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        # Add dependencies here
    ]
)"""
        
    def _generate_python_handlers(self, spec: MCPServerSpec) -> str:
        """Generate Python handler functions."""
        handlers = []
        
        for endpoint in spec.endpoints:
            handler_name = endpoint.get('name', 'handler')
            handler_code = f'''
async def {handler_name}(params: dict) -> dict:
    """Handler for {handler_name}."""
    return {{
        "success": True,
        "data": params
    }}'''
            handlers.append(handler_code)
            
        return '\n'.join(handlers)
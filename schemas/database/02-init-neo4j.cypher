// Project Aethelred - Neo4j Schema
// Version: 6.0

// Create constraints
CREATE CONSTRAINT agent_id_unique IF NOT EXISTS FOR (a:Agent) REQUIRE a.agent_id IS UNIQUE;
CREATE CONSTRAINT task_id_unique IF NOT EXISTS FOR (t:Task) REQUIRE t.task_id IS UNIQUE;
CREATE CONSTRAINT workflow_id_unique IF NOT EXISTS FOR (w:Workflow) REQUIRE w.workflow_id IS UNIQUE;

// Create indexes for performance
CREATE INDEX agent_tier_index IF NOT EXISTS FOR (a:Agent) ON (a.tier);
CREATE INDEX task_status_index IF NOT EXISTS FOR (t:Task) ON (t.status);
CREATE INDEX event_timestamp_index IF NOT EXISTS FOR (e:Event) ON (e.timestamp);

// Create initial agent nodes
CREATE (chief:Agent {
    agent_id: 'A_ChiefOfStaff_v1',
    tier: 'apex',
    role: 'System Governor',
    version: 1,
    capabilities: ['system.config.*', 'agents.evolution.*', 'rules.governance.*', 'tasks.routing'],
    is_active: true,
    created_at: datetime()
});

CREATE (auditor:Agent {
    agent_id: 'S_Auditor_v1',
    tier: 'service',
    role: 'Performance Observer',
    version: 1,
    capabilities: ['agents.observe', 'metrics.write', 'scores.calculate'],
    is_active: true,
    created_at: datetime()
});

// Create relationship between agents
MATCH (chief:Agent {agent_id: 'A_ChiefOfStaff_v1'})
MATCH (auditor:Agent {agent_id: 'S_Auditor_v1'})
CREATE (chief)-[:SUPERVISES]->(auditor);

// Create initial workflow template
CREATE (bootstrap:Workflow {
    workflow_id: 'bootstrap_system',
    name: 'System Bootstrap',
    status: 'template',
    dag: {
        nodes: [
            {id: 'init_memory', type: 'system'},
            {id: 'start_agents', type: 'agents'},
            {id: 'health_check', type: 'monitoring'}
        ],
        edges: [
            {from: 'init_memory', to: 'start_agents'},
            {from: 'start_agents', to: 'health_check'}
        ]
    },
    created_at: datetime()
});

// Create memory tier nodes
CREATE (hot:MemoryTier {
    name: 'hot',
    type: 'cache',
    backend: 'redis',
    tier_level: 0,
    max_size: '2GB',
    ttl: 3600
});

CREATE (warm:MemoryTier {
    name: 'warm',
    type: 'persistent',
    backend: 'postgresql',
    tier_level: 1,
    retention_days: 30
});

CREATE (cold:MemoryTier {
    name: 'cold',
    type: 'graph',
    backend: 'neo4j',
    tier_level: 2,
    retention_days: 365
});

CREATE (archive:MemoryTier {
    name: 'archive',
    type: 'object_store',
    backend: 'filesystem',
    tier_level: 3,
    retention_days: -1
});

// Create memory flow relationships
CREATE (hot)-[:FLOWS_TO]->(warm);
CREATE (warm)-[:FLOWS_TO]->(cold);
CREATE (cold)-[:FLOWS_TO]->(archive);

// Create capability nodes and relationships
CREATE (config_cap:Capability {name: 'system.config.*', category: 'system'});
CREATE (evolution_cap:Capability {name: 'agents.evolution.*', category: 'agents'});
CREATE (governance_cap:Capability {name: 'rules.governance.*', category: 'rules'});
CREATE (routing_cap:Capability {name: 'tasks.routing', category: 'tasks'});
CREATE (observe_cap:Capability {name: 'agents.observe', category: 'agents'});
CREATE (metrics_cap:Capability {name: 'metrics.write', category: 'metrics'});
CREATE (scores_cap:Capability {name: 'scores.calculate', category: 'scoring'});

// Link agents to their capabilities
MATCH (chief:Agent {agent_id: 'A_ChiefOfStaff_v1'})
MATCH (config_cap:Capability {name: 'system.config.*'})
MATCH (evolution_cap:Capability {name: 'agents.evolution.*'})
MATCH (governance_cap:Capability {name: 'rules.governance.*'})
MATCH (routing_cap:Capability {name: 'tasks.routing'})
CREATE (chief)-[:HAS_CAPABILITY]->(config_cap);
CREATE (chief)-[:HAS_CAPABILITY]->(evolution_cap);
CREATE (chief)-[:HAS_CAPABILITY]->(governance_cap);
CREATE (chief)-[:HAS_CAPABILITY]->(routing_cap);

MATCH (auditor:Agent {agent_id: 'S_Auditor_v1'})
MATCH (observe_cap:Capability {name: 'agents.observe'})
MATCH (metrics_cap:Capability {name: 'metrics.write'})
MATCH (scores_cap:Capability {name: 'scores.calculate'})
CREATE (auditor)-[:HAS_CAPABILITY]->(observe_cap);
CREATE (auditor)-[:HAS_CAPABILITY]->(metrics_cap);
CREATE (auditor)-[:HAS_CAPABILITY]->(scores_cap);

// Create initial governance rules
CREATE (code_standards:Rule {
    rule_id: 'code_standards_v1',
    type: 'code_standards',
    name: 'Code Quality Standards',
    version: 1,
    is_active: true,
    content: 'All code must follow PEP8 standards and include proper documentation',
    created_at: datetime()
});

CREATE (task_routing:Rule {
    rule_id: 'task_routing_v1',
    type: 'task_routing',
    name: 'Task Assignment Rules',
    version: 1,
    is_active: true,
    content: 'Tasks must be assigned based on agent capabilities and current workload',
    created_at: datetime()
});

// Link rules to governance capability
MATCH (governance_cap:Capability {name: 'rules.governance.*'})
MATCH (code_standards:Rule {rule_id: 'code_standards_v1'})
MATCH (task_routing:Rule {rule_id: 'task_routing_v1'})
CREATE (governance_cap)-[:ENFORCES]->(code_standards);
CREATE (governance_cap)-[:ENFORCES]->(task_routing);

// Create system health check query
// This will be used by monitoring agents
CREATE (:Query {
    name: 'system_health',
    cypher: 'MATCH (a:Agent) WHERE a.is_active = true RETURN count(a) as active_agents',
    category: 'monitoring',
    created_at: datetime()
});

// Create evolution tracking template
CREATE (:EvolutionTemplate {
    name: 'agent_promotion',
    criteria: {
        min_samples: 1000,
        confidence_level: 0.95,
        improvement_threshold: 0.02
    },
    shadow_duration: 86400,
    created_at: datetime()
});
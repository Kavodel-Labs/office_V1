-- Project Aethelred - PostgreSQL Schema
-- Version: 6.0

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgvector";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Create schema
CREATE SCHEMA IF NOT EXISTS aethelred;
SET search_path TO aethelred, public;

-- Enum types
CREATE TYPE agent_tier AS ENUM ('apex', 'brigade', 'doer', 'service');
CREATE TYPE task_status AS ENUM ('pending', 'assigned', 'in_progress', 'completed', 'failed', 'cancelled');
CREATE TYPE event_severity AS ENUM ('debug', 'info', 'warning', 'error', 'critical');

-- Agents table
CREATE TABLE agents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id VARCHAR(50) UNIQUE NOT NULL,
    version INTEGER NOT NULL,
    tier agent_tier NOT NULL,
    role VARCHAR(100) NOT NULL,
    capabilities JSONB NOT NULL DEFAULT '[]',
    config JSONB NOT NULL DEFAULT '{}',
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create index for agent lookups
CREATE INDEX idx_agents_agent_id_version ON agents(agent_id, version);
CREATE INDEX idx_agents_tier ON agents(tier);
CREATE INDEX idx_agents_active ON agents(is_active) WHERE is_active = true;

-- Tasks table
CREATE TABLE tasks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title VARCHAR(255) NOT NULL,
    description TEXT,
    status task_status NOT NULL DEFAULT 'pending',
    priority INTEGER DEFAULT 5 CHECK (priority BETWEEN 1 AND 10),
    assigned_to UUID REFERENCES agents(id),
    parent_task_id UUID REFERENCES tasks(id),
    workflow_id UUID,
    metadata JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    assigned_at TIMESTAMP WITH TIME ZONE,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for task queries
CREATE INDEX idx_tasks_status ON tasks(status);
CREATE INDEX idx_tasks_assigned_to ON tasks(assigned_to);
CREATE INDEX idx_tasks_workflow_id ON tasks(workflow_id);
CREATE INDEX idx_tasks_parent ON tasks(parent_task_id);

-- Events table (append-only audit log)
CREATE TABLE events (
    id BIGSERIAL PRIMARY KEY,
    event_id UUID UNIQUE DEFAULT uuid_generate_v4(),
    event_type VARCHAR(100) NOT NULL,
    severity event_severity NOT NULL DEFAULT 'info',
    agent_id UUID REFERENCES agents(id),
    task_id UUID REFERENCES tasks(id),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    data JSONB NOT NULL DEFAULT '{}',
    signature TEXT  -- Cryptographic signature
);

-- Create indexes for event queries
CREATE INDEX idx_events_timestamp ON events(timestamp);
CREATE INDEX idx_events_event_type ON events(event_type);
CREATE INDEX idx_events_agent_id ON events(agent_id);
CREATE INDEX idx_events_task_id ON events(task_id);

-- Agent performance scores table
CREATE TABLE agent_performance_scores (
    id BIGSERIAL PRIMARY KEY,
    agent_id UUID REFERENCES agents(id) NOT NULL,
    task_id UUID REFERENCES tasks(id) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    score DECIMAL(5, 4) NOT NULL CHECK (score >= 0 AND score <= 1),
    confidence_interval DECIMAL(5, 4),
    metadata JSONB NOT NULL DEFAULT '{}',
    calculated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance queries
CREATE INDEX idx_performance_agent_metric ON agent_performance_scores(agent_id, metric_name);
CREATE INDEX idx_performance_calculated_at ON agent_performance_scores(calculated_at);

-- Memory snapshots table (for time-travel debugging)
CREATE TABLE memory_snapshots (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    snapshot_type VARCHAR(50) NOT NULL,
    tier VARCHAR(20) NOT NULL,
    data JSONB NOT NULL,
    embedding vector(1536),  -- For semantic search
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create index for vector similarity search
CREATE INDEX idx_snapshots_embedding ON memory_snapshots 
    USING ivfflat (embedding vector_cosine_ops);

-- Workflows table
CREATE TABLE workflows (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    dag JSONB NOT NULL,  -- Directed Acyclic Graph definition
    status VARCHAR(50) NOT NULL DEFAULT 'created',
    initiated_by UUID REFERENCES agents(id),
    metadata JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE
);

-- Evolution tracking table
CREATE TABLE evolution_history (
    id BIGSERIAL PRIMARY KEY,
    agent_id VARCHAR(50) NOT NULL,
    from_version INTEGER NOT NULL,
    to_version INTEGER NOT NULL,
    promotion_reason TEXT,
    metrics_before JSONB NOT NULL,
    metrics_after JSONB NOT NULL,
    confidence_score DECIMAL(5, 4),
    promoted_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    promoted_by UUID REFERENCES agents(id)
);

-- Rules governance table
CREATE TABLE governance_rules (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    rule_type VARCHAR(50) NOT NULL,
    rule_name VARCHAR(100) UNIQUE NOT NULL,
    content TEXT NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    is_active BOOLEAN DEFAULT true,
    created_by UUID REFERENCES agents(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Human directives table
CREATE TABLE human_directives (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    directive_type VARCHAR(50) NOT NULL,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    success_criteria JSONB NOT NULL DEFAULT '[]',
    priority INTEGER DEFAULT 5,
    deadline TIMESTAMP WITH TIME ZONE,
    submitted_by VARCHAR(100) NOT NULL,
    assigned_to UUID REFERENCES agents(id),
    status VARCHAR(50) DEFAULT 'pending',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Task Master AI integration table
CREATE TABLE task_master_tasks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    task_id VARCHAR(255) UNIQUE NOT NULL,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    assigned_agent_id UUID REFERENCES agents(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create update timestamp trigger
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply update trigger to relevant tables
CREATE TRIGGER update_agents_updated_at BEFORE UPDATE ON agents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_tasks_updated_at BEFORE UPDATE ON tasks
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_governance_rules_updated_at BEFORE UPDATE ON governance_rules
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_task_master_tasks_updated_at BEFORE UPDATE ON task_master_tasks
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Initial data
INSERT INTO agents (agent_id, version, tier, role, capabilities) VALUES
    ('A_ChiefOfStaff', 1, 'apex', 'System Governor', 
     '["system.config.*", "agents.evolution.*", "rules.governance.*", "tasks.routing"]'::jsonb),
    ('S_Auditor', 1, 'service', 'Performance Observer',
     '["agents.observe", "metrics.write", "scores.calculate"]'::jsonb);

-- Create read-only user for monitoring
CREATE USER aethelred_monitor WITH PASSWORD 'monitor_password';
GRANT CONNECT ON DATABASE aethelred TO aethelred_monitor;
GRANT USAGE ON SCHEMA aethelred TO aethelred_monitor;
GRANT SELECT ON ALL TABLES IN SCHEMA aethelred TO aethelred_monitor;
-- Project Aethelred - Light PostgreSQL Schema
-- Optimized for lower memory usage

-- Create schema
CREATE SCHEMA IF NOT EXISTS aethelred;
SET search_path TO aethelred, public;

-- Essential tables only for basic functionality

-- Memory snapshots table (simplified)
CREATE TABLE memory_snapshots (
    id VARCHAR(255) PRIMARY KEY,  -- Changed from UUID to VARCHAR for flexible keys
    snapshot_type VARCHAR(50) NOT NULL,
    tier VARCHAR(20) NOT NULL,
    data JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Agents table (simplified)
CREATE TABLE agents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id VARCHAR(50) UNIQUE NOT NULL,
    version INTEGER NOT NULL,
    tier VARCHAR(20) NOT NULL,
    role VARCHAR(100) NOT NULL,
    capabilities JSONB NOT NULL DEFAULT '[]',
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Tasks table (simplified)
CREATE TABLE tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title VARCHAR(255) NOT NULL,
    description TEXT,
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    assigned_to UUID REFERENCES agents(id),
    metadata JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Events table (simplified)
CREATE TABLE events (
    id BIGSERIAL PRIMARY KEY,
    event_type VARCHAR(100) NOT NULL,
    agent_id UUID REFERENCES agents(id),
    task_id UUID REFERENCES tasks(id),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    data JSONB NOT NULL DEFAULT '{}'
);

-- Create essential indexes only
CREATE INDEX idx_snapshots_type_tier ON memory_snapshots(snapshot_type, tier);
CREATE INDEX idx_agents_agent_id ON agents(agent_id);
CREATE INDEX idx_tasks_status ON tasks(status);
CREATE INDEX idx_events_timestamp ON events(timestamp);

-- Insert initial agents
INSERT INTO agents (agent_id, version, tier, role, capabilities) VALUES
    ('A_ChiefOfStaff', 1, 'apex', 'System Governor', 
     '["system.config.read", "system.config.write", "tasks.routing"]'::jsonb),
    ('S_Auditor', 1, 'service', 'Performance Observer',
     '["agents.observe", "metrics.write", "scores.calculate"]'::jsonb);
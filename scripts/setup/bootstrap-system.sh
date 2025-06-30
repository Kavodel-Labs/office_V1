#!/bin/bash
# Project Aethelred - System Bootstrap Script

set -e  # Exit on error

echo "🚀 Bootstrapping Project Aethelred..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check prerequisites
check_prerequisites() {
    echo "📋 Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}❌ Docker not found. Please install Docker.${NC}"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        echo -e "${RED}❌ Docker Compose not found. Please install Docker Compose.${NC}"
        exit 1
    fi
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}❌ Python 3 not found. Please install Python 3.11+.${NC}"
        exit 1
    fi
    
    # Check Node.js
    if ! command -v node &> /dev/null; then
        echo -e "${RED}❌ Node.js not found. Please install Node.js 18+.${NC}"
        exit 1
    fi
    
    # Check task-master-ai
    if ! command -v task-master-ai &> /dev/null; then
        echo -e "${YELLOW}⚠️  task-master-ai not found globally. Installing...${NC}"
        npm install -g task-master-ai
    fi
    
    echo -e "${GREEN}✅ All prerequisites met!${NC}"
}

# Install dependencies
install_dependencies() {
    echo "📦 Installing dependencies..."
    
    # Python dependencies
    if [ ! -d "venv" ]; then
        echo "Creating Python virtual environment..."
        python3 -m venv venv
    fi
    
    echo "Activating virtual environment and installing Python packages..."
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    
    # Node.js dependencies
    echo "Installing Node.js packages..."
    npm install
    
    echo -e "${GREEN}✅ Dependencies installed!${NC}"
}

# Create required directories
create_directories() {
    echo "📁 Creating required directories..."
    
    mkdir -p logs
    mkdir -p data
    mkdir -p aethelred_archive
    mkdir -p aethelred_archive/data
    mkdir -p aethelred_archive/snapshots
    mkdir -p aethelred_archive/logs
    
    # Set permissions
    chmod 755 logs data aethelred_archive
    chmod -R 755 aethelred_archive/
    
    echo -e "${GREEN}✅ Directories created!${NC}"
}

# Start services
start_services() {
    echo "🐳 Starting Docker services..."
    
    # Check if services are already running
    if docker-compose -f config/docker-compose.dev.yml ps | grep -q "Up"; then
        echo -e "${YELLOW}⚠️  Some services already running. Stopping first...${NC}"
        docker-compose -f config/docker-compose.dev.yml down
    fi
    
    docker-compose -f config/docker-compose.dev.yml up -d
    
    # Wait for services to be healthy
    echo "⏳ Waiting for services to be healthy..."
    sleep 15
    
    # Check service health
    services=("redis" "postgres" "neo4j" "rabbitmq")
    for service in "${services[@]}"; do
        if docker-compose -f config/docker-compose.dev.yml ps | grep -q "aethelred-$service.*healthy"; then
            echo -e "${GREEN}✅ $service is healthy${NC}"
        else
            echo -e "${YELLOW}⚠️  $service may not be fully ready (will retry)${NC}"
            # Wait a bit longer for slower services
            sleep 10
        fi
    done
}

# Initialize databases
init_databases() {
    echo "🗄️  Initializing databases..."
    
    # Wait for PostgreSQL to be ready
    echo "  → Waiting for PostgreSQL..."
    timeout=60
    while [ $timeout -gt 0 ]; do
        if docker-compose -f config/docker-compose.dev.yml exec -T postgres pg_isready -U aethelred > /dev/null 2>&1; then
            break
        fi
        sleep 2
        ((timeout-=2))
    done
    
    if [ $timeout -le 0 ]; then
        echo -e "${RED}❌ PostgreSQL failed to start within timeout${NC}"
        exit 1
    fi
    
    # Initialize PostgreSQL
    echo "  → Initializing PostgreSQL schema..."
    docker-compose -f config/docker-compose.dev.yml exec -T postgres psql -U aethelred -d aethelred -f /docker-entrypoint-initdb.d/01-init.sql > /dev/null 2>&1 || true
    
    # Wait for Neo4j to be ready
    echo "  → Waiting for Neo4j..."
    timeout=60
    while [ $timeout -gt 0 ]; do
        if docker-compose -f config/docker-compose.dev.yml exec -T neo4j cypher-shell -u neo4j -p development "RETURN 1" > /dev/null 2>&1; then
            break
        fi
        sleep 2
        ((timeout-=2))
    done
    
    if [ $timeout -le 0 ]; then
        echo -e "${YELLOW}⚠️  Neo4j may not be ready yet (continuing anyway)${NC}"
    else
        # Initialize Neo4j
        echo "  → Initializing Neo4j schema..."
        docker-compose -f config/docker-compose.dev.yml exec -T neo4j cypher-shell -u neo4j -p development < schemas/database/02-init-neo4j.cypher > /dev/null 2>&1 || true
    fi
    
    echo -e "${GREEN}✅ Databases initialized!${NC}"
}

# Initialize task-master-ai
init_task_master() {
    echo "📋 Initializing Task Master AI..."
    
    # Initialize task-master-ai project if not already done
    if [ ! -d ".taskmaster" ]; then
        echo "  → Initializing task-master-ai project..."
        npx task-master-ai init --project-name "aethelred" --framework "custom" || true
    fi
    
    echo -e "${GREEN}✅ Task Master AI initialized!${NC}"
}

# Test system connectivity
test_connectivity() {
    echo "🔍 Testing system connectivity..."
    
    # Test Redis
    if docker-compose -f config/docker-compose.dev.yml exec -T redis redis-cli ping > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Redis connection: OK${NC}"
    else
        echo -e "${RED}❌ Redis connection: FAILED${NC}"
    fi
    
    # Test PostgreSQL
    if docker-compose -f config/docker-compose.dev.yml exec -T postgres psql -U aethelred -d aethelred -c "SELECT 1;" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ PostgreSQL connection: OK${NC}"
    else
        echo -e "${RED}❌ PostgreSQL connection: FAILED${NC}"
    fi
    
    # Test Neo4j
    if docker-compose -f config/docker-compose.dev.yml exec -T neo4j cypher-shell -u neo4j -p development "RETURN 1" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Neo4j connection: OK${NC}"
    else
        echo -e "${YELLOW}⚠️  Neo4j connection: May need more time${NC}"
    fi
    
    # Test RabbitMQ
    if docker-compose -f config/docker-compose.dev.yml exec -T rabbitmq rabbitmq-diagnostics ping > /dev/null 2>&1; then
        echo -e "${GREEN}✅ RabbitMQ connection: OK${NC}"
    else
        echo -e "${YELLOW}⚠️  RabbitMQ connection: May need more time${NC}"
    fi
}

# Main execution
main() {
    echo "======================================"
    echo "   PROJECT AETHELRED BOOTSTRAP"
    echo "======================================"
    echo ""
    
    # Check if running from correct directory
    if [ ! -f "config/aethelred-config.yaml" ]; then
        echo -e "${RED}❌ Please run this script from the aethelred project root directory${NC}"
        exit 1
    fi
    
    check_prerequisites
    install_dependencies
    create_directories
    start_services
    init_databases
    init_task_master
    test_connectivity
    
    echo ""
    echo -e "${GREEN}🎉 Bootstrap complete!${NC}"
    echo ""
    echo -e "${BLUE}📊 Service URLs:${NC}"
    echo "  - Redis:      localhost:6379"
    echo "  - PostgreSQL: localhost:5432"
    echo "  - Neo4j:      http://localhost:7474"
    echo "  - RabbitMQ:   http://localhost:15672"
    echo "  - Prometheus: http://localhost:9090"
    echo "  - Grafana:    http://localhost:3000"
    echo "  - Jaeger:     http://localhost:16686"
    echo ""
    echo -e "${BLUE}🚀 To start Aethelred:${NC}"
    echo "  1. Activate Python environment: source venv/bin/activate"
    echo "  2. Run the system: python main.py"
    echo ""
    echo -e "${BLUE}🧪 To run tests:${NC}"
    echo "  1. pytest tests/"
    echo ""
    echo -e "${BLUE}📋 Task Master AI commands:${NC}"
    echo "  - npm run task-list    # List tasks"
    echo "  - npm run task-status  # Check task status"
    echo "  - npm run task         # Interactive task management"
    echo ""
    
    # Check if user wants to start Aethelred immediately
    read -p "Start Aethelred system now? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${BLUE}🚀 Starting Aethelred...${NC}"
        source venv/bin/activate
        python main.py
    fi
}

# Run main function
main "$@"
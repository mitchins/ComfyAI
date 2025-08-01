#!/bin/bash
# ComfyAI Deployment Script
# Provides easy commands to build and run the unified ComfyAI server

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    echo "ComfyAI Deployment Script"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  build     Build the Docker image"
    echo "  run       Run the server using Docker Compose"
    echo "  stop      Stop the running server"
    echo "  restart   Restart the server"
    echo "  logs      Show server logs"
    echo "  dev       Run in development mode (with file watching)"
    echo "  test      Run the test suite"
    echo "  clean     Clean up Docker images and volumes"
    echo "  help      Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 build         # Build the Docker image"
    echo "  $0 run           # Start the server on http://localhost:8000"
    echo "  $0 logs          # View real-time logs"
    echo ""
}

# Function to check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker and try again."
        exit 1
    fi
}

# Function to build the Docker image
build_image() {
    print_status "Building ComfyAI Docker image..."
    docker build -t comfyai:latest .
    print_success "Docker image built successfully!"
}

# Function to run the server
run_server() {
    print_status "Starting ComfyAI server..."
    docker-compose up -d
    print_success "Server started! Available at:"
    echo "  🏠 Homepage: http://localhost:8000/"
    echo "  🌐 Management UI: http://localhost:8000/ui"
    echo "  🎯 Vision Test: http://localhost:8000/test"
    echo "  🔗 API Docs: http://localhost:8000/docs"
    echo ""
    print_status "Use '$0 logs' to view server logs"
}

# Function to stop the server
stop_server() {
    print_status "Stopping ComfyAI server..."
    docker-compose down
    print_success "Server stopped!"
}

# Function to restart the server
restart_server() {
    print_status "Restarting ComfyAI server..."
    docker-compose restart
    print_success "Server restarted!"
}

# Function to show logs
show_logs() {
    print_status "Showing server logs (Ctrl+C to exit)..."
    docker-compose logs -f
}

# Function to run in development mode
dev_mode() {
    print_status "Starting ComfyAI in development mode..."
    print_warning "This will bind-mount the current directory for live updates"
    
    # Create a development docker-compose override
    cat > docker-compose.override.yml << EOF
version: '3.8'
services:
  comfyai:
    volumes:
      - .:/app
    environment:
      - PYTHONUNBUFFERED=1
      - PYTHONDONTWRITEBYTECODE=1
      - RELOAD=true
    command: ["uvicorn", "apps.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
EOF
    
    docker-compose up
}

# Function to run tests
run_tests() {
    print_status "Running test suite..."
    if command -v python3 &> /dev/null; then
        python3 -m pytest tests/ -v
    else
        print_warning "Python not found locally, running tests in Docker..."
        docker run --rm -v "$(pwd):/app" -w /app comfyai:latest python -m pytest tests/ -v
    fi
}

# Function to clean up
cleanup() {
    print_status "Cleaning up Docker resources..."
    docker-compose down -v
    docker image prune -f
    docker volume prune -f
    print_success "Cleanup completed!"
}

# Main command handling
case "${1:-help}" in
    build)
        check_docker
        build_image
        ;;
    run)
        check_docker
        run_server
        ;;
    stop)
        check_docker
        stop_server
        ;;
    restart)
        check_docker
        restart_server
        ;;
    logs)
        check_docker
        show_logs
        ;;
    dev)
        check_docker
        dev_mode
        ;;
    test)
        run_tests
        ;;
    clean)
        check_docker
        cleanup
        ;;
    help|--help|-h)
        show_usage
        ;;
    *)
        print_error "Unknown command: $1"
        echo ""
        show_usage
        exit 1
        ;;
esac
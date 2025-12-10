.PHONY: install start stop clean

# Install dependencies
install:
	@echo "Installing backend dependencies..."
	cd backend && pip3 install -r requirements.txt
	@echo "Installing frontend dependencies..."
	cd frontend && npm install
	@echo "✓ Installation complete!"

# Start both services
start:
	@echo "Starting GLM-ASR Web..."
	./start.sh

# Stop services (find and kill processes)
stop:
	@echo "Stopping services..."
	@pkill -f "python3 main.py" || true
	@pkill -f "npm run dev" || true
	@echo "✓ Services stopped"

# Clean temporary files
clean:
	@echo "Cleaning temporary files..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "node_modules" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".next" -exec rm -rf {} + 2>/dev/null || true
	@echo "✓ Clean complete"

# Development setup (install + start)
dev: install start

# Show help
help:
	@echo "GLM-ASR Web - Makefile Commands"
	@echo ""
	@echo "  make install  - Install all dependencies"
	@echo "  make start    - Start frontend and backend"
	@echo "  make stop     - Stop all services"
	@echo "  make clean    - Remove temporary files"
	@echo "  make dev      - Install and start (first time setup)"
	@echo "  make help     - Show this help message"

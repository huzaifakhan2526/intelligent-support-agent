.PHONY: help install install-dev test test-cov clean format lint type-check setup run

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install production dependencies
	pip install -r requirements.txt

install-dev: ## Install development dependencies
	pip install -r requirements-dev.txt

test: ## Run tests
	pytest tests/ -v

test-cov: ## Run tests with coverage
	pytest tests/ --cov=src --cov-report=html --cov-report=term-missing

clean: ## Clean up build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
   rm -rf .coverage
   rm -rf htmlcov/
   find . -type d -name __pycache__ -delete
   find . -type f -name "*.pyc" -delete

format: ## Format code with black and isort
   black src/ tests/
   isort src/ tests/

lint: ## Lint code with flake8
   flake8 src/ tests/

type-check: ## Type check with mypy
   mypy src/

setup: ## Initial project setup
   python -m venv venv
   @echo "Virtual environment created. Activate with:"
   @echo "  source venv/bin/activate  (Linux/Mac)"
   @echo "  venv\\Scripts\\activate     (Windows)"
   @echo "Then run: make install-dev"

run: ## Run the chatbot (command line)
   python src/chatbot.py

run-ws: ## Run the WebSocket server
   python src/websocket_server.py

run-web: ## Run the web interface with HTTP server
   python src/http_server.py

run-prod: ## Run with production config
   python src/chatbot.py --config configs/config_production.yaml

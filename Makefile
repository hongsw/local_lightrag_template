.PHONY: help install dev test lint clean docker-build docker-run docker-stop docker-logs k8s-deploy k8s-delete

# Variables
IMAGE_NAME := lightrag-api
IMAGE_TAG := latest
CONTAINER_NAME := lightrag-api

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ========================
# Development
# ========================

install:  ## Install dependencies
	pip install -r requirements.txt

dev:  ## Run development server
	python run.py

test:  ## Run tests
	pytest tests/ -v

lint:  ## Run linter
	ruff check src/

clean:  ## Clean cache files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov

# ========================
# Docker
# ========================

docker-build:  ## Build Docker image
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) .

docker-run:  ## Run Docker container
	docker compose up -d

docker-stop:  ## Stop Docker container
	docker compose down

docker-logs:  ## View Docker logs
	docker compose logs -f

docker-shell:  ## Shell into container
	docker exec -it $(CONTAINER_NAME) /bin/bash

docker-clean:  ## Remove Docker image and volumes
	docker compose down -v
	docker rmi $(IMAGE_NAME):$(IMAGE_TAG) || true

# ========================
# Kubernetes
# ========================

k8s-deploy:  ## Deploy to Kubernetes
	kubectl apply -k k8s/

k8s-delete:  ## Delete from Kubernetes
	kubectl delete -k k8s/

k8s-status:  ## Check Kubernetes status
	kubectl -n lightrag get all

k8s-logs:  ## View Kubernetes logs
	kubectl -n lightrag logs -f deployment/lightrag-api

k8s-port-forward:  ## Port forward to local
	kubectl -n lightrag port-forward svc/lightrag-api 8000:80

# ========================
# Index Operations
# ========================

index-local:  ## Index local PDF files
	curl -X POST "http://localhost:8000/index/local" \
		-H "Content-Type: application/json" \
		-d '{"path": "/app/rag_raw_pdfs", "recursive": true}'

clear-index:  ## Clear all indexed data
	curl -X DELETE "http://localhost:8000/index/clear"

health:  ## Check health
	curl -s "http://localhost:8000/health" | python -m json.tool

stats:  ## Get stats
	curl -s "http://localhost:8000/stats" | python -m json.tool

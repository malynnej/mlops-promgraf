# Use Docker Compose v2
COMPOSE=docker compose

.PHONY: all stop evaluation fire-alert

# Start all services (API, Prometheus, Grafana, Node Exporter)
all:
	$(COMPOSE) up -d

# Stop all services
stop:
	$(COMPOSE) down

# Run run_evaluation.py once inside the evaluation container
evaluation:
	$(COMPOSE) run --rm evaluation python run_evaluation.py

# Intentionally trigger an alert (BikeApiDown)
fire-alert:
	# Stop the API so the BikeApiDown alert fires
	$(COMPOSE) stop bike-api
	@echo "bike-api stopped. Wait a couple of minutes and check Prometheus Alerts UI."
	@echo "To restore the system, run: make all"

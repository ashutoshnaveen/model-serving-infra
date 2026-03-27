.PHONY: serve dev test bench clean setup

# Start production server
serve:
	uvicorn src.api.server:app --host 0.0.0.0 --port 8000

# Start dev server with auto-reload
dev:
	uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload

# Run tests
test:
	pytest tests/ -v

# Run benchmarks
bench:
	python -m benchmarks.run_benchmark

# Download default model
download-model:
	python -m src.models.loader --model gpt2

# Clean caches
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache

# Initial setup
setup:
	python -m venv venv
	. venv/bin/activate && pip install -r requirements.txt
	@echo "Run 'source venv/bin/activate' to activate the virtual environment"

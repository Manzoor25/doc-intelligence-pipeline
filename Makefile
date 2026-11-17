.PHONY: up down dataset demo demos-failures eval eval-ci test lint logs

COMPOSE ?= docker compose

up:
	$(COMPOSE) up -d --build --wait
	@echo "Stack ready -> Review UI: http://localhost:8000 | API Docs: http://localhost:8000/docs | MinIO Console: http://localhost:9001"

dataset:
	$(COMPOSE) run --rm --no-deps api python -m doc_pipeline.synthetic.generator --count 200 --seed 42

demo: up
	$(COMPOSE) exec -T api python -m doc_pipeline.synthetic.generator --count 200 --seed 42 --skip-if-exists
	$(COMPOSE) exec -T api python -m scripts.run_end_to_end_demo
	$(COMPOSE) exec -T api python -m scripts.demos.demo_rate_limit
	$(COMPOSE) exec -T api python -m scripts.demos.demo_malformed_json
	$(COMPOSE) exec -T api python -m scripts.demos.demo_prompt_injection
	$(COMPOSE) exec -T api python -m scripts.demos.demo_model_swap

eval: up
	$(COMPOSE) exec -T api python -m doc_pipeline.synthetic.generator --count 200 --seed 42 --skip-if-exists
	$(COMPOSE) exec -T api python -m doc_pipeline.eval.runner --subset benchmark --compare-ocr

eval-ci:
	python -m doc_pipeline.synthetic.generator --count 200 --seed 42 --skip-if-exists
	python -m doc_pipeline.eval.runner --subset ci --in-process-mock --fail-on-regression

test:
	$(COMPOSE) run --rm --no-deps api sh -c "ruff check src tests scripts && pytest -v"

down:
	$(COMPOSE) down -v --remove-orphans

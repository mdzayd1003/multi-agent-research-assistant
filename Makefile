.PHONY: install test quick ingest ask serve docker clean

install:
	pip install -r requirements.txt

test:
	python -m pytest -q

# End to end on the sample corpus with the deterministic backend: no model, no network.
quick: ingest
	python -m research_assistant --offline ask "How does top-k retrieval work in this project?" \
		--fake-llm --trace results/smoke/trace.jsonl

ingest:
	python -m research_assistant ingest sample_corpus

ask:
	python -m research_assistant ask "$(Q)"

serve:
	python -m research_assistant serve

docker:
	docker compose up --build

clean:
	rm -rf .cache results/traces .pytest_cache

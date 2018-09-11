TEAM ?=
TAG ?= practice
P ?= "Sample Bot"
N ?= 99
T ?= 12345678
IP ?= localhost
PO ?=  3333


.PHONY: clean_cache
clean_cache:
	find . -name "*.pyc" -exec rm --force {} +
	find . -name "__pycache__" -exec rm -rf {} + 


.PHONY: build_image
build_image: clean_cache
	docker build -t trend-hearts . 


.PHONY: run
run: build_image
	docker run -v $(PWD)/log:/log trend-hearts $(P) $(N) $(T) ws://$(IP):$(PO)


.PHONY: tag
tag:
	docker tag trend-hearts ai.registry.trendmicro.com/$(TEAM)/trend-hearts:$(TAG)


.PHONY: tag_all
tag_all:
	docker tag trend-hearts ai.registry.trendmicro.com/$(TEAM)/trend-hearts:practice
	docker tag trend-hearts ai.registry.trendmicro.com/$(TEAM)/trend-hearts:rank


.PHONY: push
push:
	docker login ai.registry.trendmicro.com
	docker push ai.registry.trendmicro.com/$(TEAM)/trend-hearts:$(TAG)


.PHONY: push_all
push_all:
	docker login ai.registry.trendmicro.com
	docker push ai.registry.trendmicro.com/$(TEAM)/trend-hearts:practice
	docker push ai.registry.trendmicro.com/$(TEAM)/trend-hearts:rank


.PHONY: build
build: build_image tag_all push_all

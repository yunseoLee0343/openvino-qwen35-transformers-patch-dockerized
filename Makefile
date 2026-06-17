IMAGE ?= qwen35-ov-use-cache-repro:local
REPORT_DIR ?= $(PWD)/reports
MODEL_ID ?= Qwen/Qwen3.5-0.8B
OPTIMUM_INTEL_REF ?= v2.0.0-release
TRANSFORMERS_REF ?= v5.11.0
OPENVINO_VERSION ?= 2026.2.0
RUN_FULL_EXPORT ?= 1

.PHONY: build run clean

build:
	docker build --build-arg OPENVINO_VERSION=$(OPENVINO_VERSION) --build-arg TRANSFORMERS_REF=$(TRANSFORMERS_REF) -t $(IMAGE) .

run:
	mkdir -p $(REPORT_DIR)
	docker run --rm \
		-e MODEL_ID="$(MODEL_ID)" \
		-e OPTIMUM_INTEL_REF="$(OPTIMUM_INTEL_REF)" \
		-e TRANSFORMERS_REF="$(TRANSFORMERS_REF)" \
		-e OPENVINO_VERSION="$(OPENVINO_VERSION)" \
		-e RUN_FULL_EXPORT="$(RUN_FULL_EXPORT)" \
		-v "$(REPORT_DIR):/work/reports" \
		-v "$(HOME)/.cache/huggingface:/root/.cache/huggingface" \
		$(IMAGE)

clean:
	rm -rf reports/*

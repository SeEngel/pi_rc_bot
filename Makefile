SHELL := /bin/bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c

# Root of this repo (absolute path, no trailing slash)
PROJECT_ROOT := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))

# Creates a venv folder under .venv with the name "agent_venv"
VENV_DIR ?= $(PROJECT_ROOT)/.venv/agent_venv
AGENT_REQUIREMENTS ?= $(PROJECT_ROOT)/agent/requirements.txt

.PHONY: help uv agent-venv agent-deps agent-setup clean-agent-venv

help:
	@printf '%s\n' \
		"Targets:" \
		"  make uv             Install uv (if missing)" \
		"  make agent-venv      Create venv at $(VENV_DIR) with Python 3.12" \
		"  make agent-deps      Install $(AGENT_REQUIREMENTS) into that venv" \
		"  make agent-setup     uv + venv + deps" \
		"  make clean-agent-venv Remove $(VENV_DIR)"

uv:
	# 1) install uv
	UV="$$(command -v uv 2>/dev/null || true)"
	if [[ -z "$$UV" && -x "$$HOME/.local/bin/uv" ]]; then UV="$$HOME/.local/bin/uv"; fi
	if [[ -z "$$UV" && -x "$$HOME/.cargo/bin/uv" ]]; then UV="$$HOME/.cargo/bin/uv"; fi
	if [[ -n "$$UV" ]]; then
		echo "uv already installed: $$UV"
		exit 0
	fi

	echo "Installing uv via https://astral.sh/uv/install.sh ..."
	curl -LsSf https://astral.sh/uv/install.sh | sh

	UV="$$(command -v uv 2>/dev/null || true)"
	if [[ -z "$$UV" && -x "$$HOME/.local/bin/uv" ]]; then UV="$$HOME/.local/bin/uv"; fi
	if [[ -z "$$UV" && -x "$$HOME/.cargo/bin/uv" ]]; then UV="$$HOME/.cargo/bin/uv"; fi
	if [[ -z "$$UV" ]]; then
		echo "uv installed, but not found on PATH." >&2
		echo "Try: export PATH=\"$$HOME/.local/bin:$$HOME/.cargo/bin:$$PATH\"" >&2
		exit 1
	fi
	echo "uv installed: $$UV"

agent-venv: uv
	# 2) create venv (Python 3.12)
	UV="$$(command -v uv 2>/dev/null || true)"
	if [[ -z "$$UV" && -x "$$HOME/.local/bin/uv" ]]; then UV="$$HOME/.local/bin/uv"; fi
	if [[ -z "$$UV" && -x "$$HOME/.cargo/bin/uv" ]]; then UV="$$HOME/.cargo/bin/uv"; fi
	if [[ -z "$$UV" ]]; then echo "uv not found; run 'make uv'" >&2; exit 1; fi

	mkdir -p "$(dir $(VENV_DIR))"
	echo "Creating venv at $(VENV_DIR) (Python 3.12) ..."
	"$$UV" venv "$(VENV_DIR)" --python 3.12
	echo "Venv ready: $(VENV_DIR)"

agent-deps: agent-venv
	# 3) install only agent requirements into the venv
	if [[ ! -f "$(AGENT_REQUIREMENTS)" ]]; then
		echo "Missing requirements file: $(AGENT_REQUIREMENTS)" >&2
		exit 1
	fi
	if [[ ! -x "$(VENV_DIR)/bin/python" ]]; then
		echo "Venv python not found: $(VENV_DIR)/bin/python" >&2
		echo "Run: make agent-venv" >&2
		exit 1
	fi

	UV="$$(command -v uv 2>/dev/null || true)"
	if [[ -z "$$UV" && -x "$$HOME/.local/bin/uv" ]]; then UV="$$HOME/.local/bin/uv"; fi
	if [[ -z "$$UV" && -x "$$HOME/.cargo/bin/uv" ]]; then UV="$$HOME/.cargo/bin/uv"; fi
	if [[ -z "$$UV" ]]; then echo "uv not found; run 'make uv'" >&2; exit 1; fi

	echo "Installing agent deps from $(AGENT_REQUIREMENTS) ..."
	"$$UV" pip install --python "$(VENV_DIR)/bin/python" -r "$(AGENT_REQUIREMENTS)"
	echo "Done."

agent-setup: uv agent-venv agent-deps

clean-agent-venv:
	rm -rf "$(VENV_DIR)"
	echo "Removed $(VENV_DIR)"

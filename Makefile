# Makefile for Voice Cloning Project
# ----------------------------------
# Provides handy commands for creating/updating a virtual environment,
# installing requirements, and running common project scripts.
#
# Usage examples:
#   make create-env
#   make install-requirements
#   make update-requirements
#   make data-preprocess
#   make train-encoder
#   make train-synthesizer
#   make train-vocoder
#   make run-mlflow-server
#   ... and more.

# Name (or path) of the virtual environment directory
VENV_DIR = .venv
PYTHON   = $(VENV_DIR)/bin/python
PIP      = $(VENV_DIR)/bin/pip

# Default shell
SHELL = /bin/bash

.PHONY: help create-env install-requirements update-requirements \
        data-preprocess train-encoder train-synthesizer train-vocoder \
        run-mlflow-server run-wandb-sweep clean

## help: Show all available make targets.
help:
	@echo "Available make targets:"
	@echo "  make create-env            - Create a new Python virtual environment in $(VENV_DIR)"
	@echo "  make install-requirements  - Install packages from requirements.txt into $(VENV_DIR)"
	@echo "  make update-requirements   - Update installed packages (reinstall from requirements.txt)"
	@echo "  make data-preprocess       - Run data preprocessing script"
	@echo "  make train-encoder         - Train speaker encoder module (MLflow entry point)"
	@echo "  make train-synthesizer     - Train synthesizer module (MLflow entry point)"
	@echo "  make train-vocoder         - Train vocoder module (MLflow entry point)"
	@echo "  make run-mlflow-server     - Launch a local MLflow tracking server"
	@echo "  make run-wandb-sweep       - Launch a W&B hyperparameter sweep"
	@echo "  make clean                 - Remove the virtual environment and cached files"
	@echo "  make help                  - Show this message"

## create-env: Create a Python virtual environment in .venv
create-env:
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo ">>> Creating virtual environment in $(VENV_DIR)"; \
		python -m venv $(VENV_DIR); \
		echo ">>> Virtual environment created."; \
	else \
		echo ">>> $(VENV_DIR) already exists. Skipping creation."; \
	fi

## install-requirements: Install packages from requirements.txt
install-requirements: create-env
	@echo ">>> Installing requirements..."
	@$(PIP) install --upgrade pip
	@$(PIP) install -r requirements.txt
	@echo ">>> Installation complete."

## update-requirements: Reinstall or update packages from requirements.txt
update-requirements: create-env
	@echo ">>> Updating requirements..."
	@$(PIP) install --upgrade pip
	@$(PIP) install --upgrade -r requirements.txt
	@echo ">>> Update complete."

## data-preprocess: Run data preprocessing script
data-preprocess: create-env
	@echo ">>> Running data preprocessing..."
	@. $(VENV_DIR)/bin/activate; \
		./scripts/run_data_preprocessing.sh
	@echo ">>> Data preprocessing finished."

## train-encoder: Train the speaker encoder module (MLflow entry point)
train-encoder: create-env
	@echo ">>> Training speaker encoder..."
	@. $(VENV_DIR)/bin/activate; \
		./scripts/run_encoder_training.sh
	@echo ">>> Encoder training completed."

## train-synthesizer: Train the synthesizer module
train-synthesizer: create-env
	@echo ">>> Training synthesizer..."
	@. $(VENV_DIR)/bin/activate; \
		./scripts/run_synthesizer_training.sh
	@echo ">>> Synthesizer training completed."

## train-vocoder: Train the vocoder module
train-vocoder: create-env
	@echo ">>> Training vocoder..."
	@. $(VENV_DIR)/bin/activate; \
		./scripts/run_vocoder_training.sh
	@echo ">>> Vocoder training completed."

## run-mlflow-server: Launch a local MLflow tracking server
run-mlflow-server: create-env
	@echo ">>> Starting MLflow server..."
	@. $(VENV_DIR)/bin/activate; \
		./scripts/run_mlflow_server.sh

## run-wandb-sweep: Launch a W&B hyperparameter sweep
run-wandb-sweep: create-env
	@echo ">>> Launching W&B sweep..."
	@. $(VENV_DIR)/bin/activate; \
		./scripts/run_wandb_sweep.sh

## clean: Remove the virtual environment and any cache artifacts
clean:
	@echo ">>> Cleaning up..."
	@rm -rf $(VENV_DIR)
	@find . -name "__pycache__" -type d -exec rm -rf {} +
	@echo ">>> Clean complete."

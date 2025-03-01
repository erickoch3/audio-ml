#!/usr/bin/env bash
# Purpose: Launches a local MLFlow tracking server.
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns

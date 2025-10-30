#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PROJECT=""
REGION="us-central1"
SERVICE="trading-llm-api"
IMAGE=""
PORT="8080"
PLATFORM="managed"
ALLOW_UNAUTH=true
SKIP_BUILD=false
RUN_TESTS=false
TEST_CMD=("pytest" "trading_llm/tests/test_api.py")
ENV_FILE=""
ENV_VAR_ITEMS=()
EXTRA_DEPLOY_ARGS=()

usage() {
  cat <<'EOF'
Usage: cloud_run_deploy.sh [options] [-- <extra gcloud args>]

Automates building and deploying the Trading LLM API to Cloud Run.

Options:
  -p, --project ID          Google Cloud project ID (defaults to gcloud config)
  -r, --region REGION       Cloud Run region (default: us-central1)
  -s, --service NAME        Cloud Run service name (default: trading-llm-api)
  -i, --image IMAGE         Fully-qualified container image (default: gcr.io/<project>/<service>)
  -e, --env-file PATH       Path to env file containing KEY=VALUE pairs for --set-env-vars
      --port PORT           Container port exposed in Cloud Run (default: 8080)
      --no-allow-unauth     Disable unauthenticated access (default: allow)
      --skip-build          Skip gcloud builds submit (assumes image already pushed)
      --run-tests           Run pytest trading_llm/tests/test_api.py before building
      --test-cmd CMD...     Custom test command (provide after flag, repeat args as needed)
  -h, --help                Show this help message

Any arguments following "--" are passed directly to "gcloud run deploy".
EOF
}

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

check_command() {
  command -v "$1" >/dev/null 2>&1 || die "Required command '$1' not found in PATH"
}

load_env_file() {
  local file="$1"
  [[ -f "$file" ]] || die "Env file '$file' does not exist"

  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue

    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"

    [[ -z "$line" ]] && continue

    if [[ "$line" == export* ]]; then
      line="${line#export }"
      line="${line#"${line%%[![:space:]]*}"}"
    fi
    if [[ "$line" != *"="* ]]; then
      die "Invalid env entry in '$file': $line"
    fi
    ENV_VAR_ITEMS+=("${line}")
  done <"$file"
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -p|--project)
        PROJECT="$2"
        shift 2
        ;;
      -r|--region)
        REGION="$2"
        shift 2
        ;;
      -s|--service)
        SERVICE="$2"
        shift 2
        ;;
      -i|--image)
        IMAGE="$2"
        shift 2
        ;;
      -e|--env-file)
        ENV_FILE="$2"
        shift 2
        ;;
      --port)
        PORT="$2"
        shift 2
        ;;
      --no-allow-unauth)
        ALLOW_UNAUTH=false
        shift
        ;;
      --skip-build)
        SKIP_BUILD=true
        shift
        ;;
      --run-tests)
        RUN_TESTS=true
        shift
        ;;
      --test-cmd)
        TEST_CMD=()
        shift
        while [[ $# -gt 0 ]]; do
          [[ "$1" == "--" ]] && break
          TEST_CMD+=("$1")
          shift
        done
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      --)
        shift
        EXTRA_DEPLOY_ARGS+=("$@")
        break
        ;;
      *)
        die "Unknown option: $1"
        ;;
    esac
  done
}

main() {
  parse_args "$@"

  check_command gcloud

  if [[ ${RUN_TESTS} == true ]]; then
    check_command "${TEST_CMD[0]}"
  fi

  if [[ -z "$PROJECT" ]]; then
    PROJECT="$(gcloud config get-value project 2>/dev/null || true)"
  fi

  [[ -n "$PROJECT" ]] || die "Project ID not provided and not set in gcloud config"

  if [[ -n "$ENV_FILE" ]]; then
    load_env_file "$ENV_FILE"
  fi

  if [[ -z "$IMAGE" ]]; then
    IMAGE="gcr.io/${PROJECT}/${SERVICE}"
  fi

  echo "[INFO] Using project: ${PROJECT}"
  echo "[INFO] Region: ${REGION}"
  echo "[INFO] Service: ${SERVICE}"
  echo "[INFO] Image: ${IMAGE}"
  echo "[INFO] Port: ${PORT}"
  if [[ ${RUN_TESTS} == true ]]; then
    echo "[INFO] Test command: ${TEST_CMD[*]}"
  fi
  if [[ ${ALLOW_UNAUTH} == true ]]; then
    echo "[INFO] Unauthenticated access: allowed"
  else
    echo "[INFO] Unauthenticated access: disabled"
  fi
  if [[ ${SKIP_BUILD} == true ]]; then
    echo "[INFO] Skipping build step"
  fi
  if [[ ${#ENV_VAR_ITEMS[@]} -gt 0 ]]; then
    echo "[INFO] Loaded ${#ENV_VAR_ITEMS[@]} env vars from ${ENV_FILE}"
  fi
  if [[ ${#EXTRA_DEPLOY_ARGS[@]} -gt 0 ]]; then
    echo "[INFO] Extra deploy args: ${EXTRA_DEPLOY_ARGS[*]}"
  fi

  pushd "${PROJECT_ROOT}" >/dev/null

  if [[ ${RUN_TESTS} == true ]]; then
    echo "[INFO] Running tests before deployment"
    "${TEST_CMD[@]}"
  fi

  if [[ ${SKIP_BUILD} == false ]]; then
    echo "[INFO] Submitting build to Cloud Build"
    gcloud builds submit --tag "${IMAGE}"
  fi

  DEPLOY_ARGS=(
    "gcloud" "run" "deploy" "${SERVICE}"
    "--image" "${IMAGE}"
    "--platform" "${PLATFORM}"
    "--region" "${REGION}"
    "--port" "${PORT}"
  )

  if [[ ${ALLOW_UNAUTH} == true ]]; then
    DEPLOY_ARGS+=("--allow-unauthenticated")
  else
    DEPLOY_ARGS+=("--no-allow-unauthenticated")
  fi

  if [[ ${#ENV_VAR_ITEMS[@]} -gt 0 ]]; then
    local joined
    joined=$(IFS=,; printf '%s' "${ENV_VAR_ITEMS[*]}")
    DEPLOY_ARGS+=("--set-env-vars" "${joined}")
  fi

  if [[ ${#EXTRA_DEPLOY_ARGS[@]} -gt 0 ]]; then
    DEPLOY_ARGS+=("${EXTRA_DEPLOY_ARGS[@]}")
  fi

  echo "[INFO] Deploying to Cloud Run"
  "${DEPLOY_ARGS[@]}"

  popd >/dev/null

  echo "[INFO] Deployment complete"
}

main "$@"


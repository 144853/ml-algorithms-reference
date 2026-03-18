#!/usr/bin/env bash
#
# Deploy any algorithm to Google Cloud Vertex AI.
#
# Usage:
#   ./deploy.sh linear_regression_sklearn dev my-gcp-project
#   ./deploy.sh random_forest_sklearn prod my-gcp-project
#   ./deploy.sh arima_sklearn staging my-gcp-project
#
# To deploy a DIFFERENT algorithm, just change the first argument.

set -euo pipefail

ALGORITHM_NAME="${1:?Usage: ./deploy.sh <algorithm_name> <environment> <project_id>}"
ENVIRONMENT="${2:-dev}"
PROJECT_ID="${3:?GCP project ID required}"
REGION="${GCP_REGION:-us-central1}"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

echo "=== Deploying ${ALGORITHM_NAME} to Vertex AI (${ENVIRONMENT}) ==="

# --- Step 1: Find algorithm ---
find_algorithm() {
    local name="$1"
    local found
    found=$(find "${REPO_ROOT}" -name "${name}.py" -not -path "*/mlops/*" -not -path "*/.claude/*" | head -1)
    if [[ -z "$found" ]]; then
        echo "ERROR: Algorithm '${name}' not found" >&2
        exit 1
    fi
    echo "${found#${REPO_ROOT}/}"
}

ALGORITHM_PATH=$(find_algorithm "${ALGORITHM_NAME}")
echo "Found algorithm at: ${ALGORITHM_PATH}"

# --- Step 2: Train locally ---
echo "=== Training model locally ==="
MODEL_DIR="/tmp/ml-models/${ALGORITHM_NAME}"
mkdir -p "${MODEL_DIR}"

cd "${REPO_ROOT}"
python mlops/common/train_wrapper.py \
    --algorithm-path "${ALGORITHM_PATH}" \
    --output-dir "${MODEL_DIR}"

# --- Step 3: Upload model artifacts to GCS ---
echo "=== Uploading model to GCS ==="
BUCKET="${PROJECT_ID}-ml-models-${ENVIRONMENT}"
gsutil cp "${MODEL_DIR}/model.pkl" "gs://${BUCKET}/${ALGORITHM_NAME}/model.pkl"
gsutil cp "${MODEL_DIR}/metadata.json" "gs://${BUCKET}/${ALGORITHM_NAME}/metadata.json"

# --- Step 4: Build and push container ---
echo "=== Building container ==="
AR_REPO="${REGION}-docker.pkg.dev/${PROJECT_ID}/ml-algorithms-${ENVIRONMENT}"

gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

docker build -t "${AR_REPO}/ml-algorithms:${ALGORITHM_NAME}-latest" \
    -f mlops/common/Dockerfile \
    "${REPO_ROOT}"

docker push "${AR_REPO}/ml-algorithms:${ALGORITHM_NAME}-latest"

# --- Step 5: Deploy with Terraform ---
echo "=== Deploying infrastructure with Terraform ==="
cd "${REPO_ROOT}/mlops/google_ml/terraform"

terraform init -backend-config="prefix=vertex-ai/${ALGORITHM_NAME}/${ENVIRONMENT}"

terraform apply \
    -var="project_id=${PROJECT_ID}" \
    -var="algorithm_name=${ALGORITHM_NAME}" \
    -var="environment=${ENVIRONMENT}" \
    -var="region=${REGION}" \
    -auto-approve

# --- Step 6: Test endpoint ---
echo "=== Testing endpoint ==="
ENDPOINT_ID=$(terraform output -raw endpoint_id)

gcloud ai endpoints predict "${ENDPOINT_ID}" \
    --project="${PROJECT_ID}" \
    --region="${REGION}" \
    --json-request='{"instances": [[1.0, 2.0, 3.0, 4.0, 5.0]]}'

echo "=== Deployment complete ==="

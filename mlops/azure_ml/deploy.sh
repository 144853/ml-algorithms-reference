#!/usr/bin/env bash
#
# Deploy any algorithm to Azure Machine Learning.
#
# Usage:
#   ./deploy.sh linear_regression_sklearn dev
#   ./deploy.sh random_forest_sklearn prod
#   ./deploy.sh arima_sklearn staging
#
# To deploy a DIFFERENT algorithm, just change the first argument.

set -euo pipefail

ALGORITHM_NAME="${1:?Usage: ./deploy.sh <algorithm_name> <environment>}"
ENVIRONMENT="${2:-dev}"
LOCATION="${AZURE_LOCATION:-eastus}"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

echo "=== Deploying ${ALGORITHM_NAME} to Azure ML (${ENVIRONMENT}) ==="

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

# --- Step 3: Build and push container ---
echo "=== Building container ==="
RESOURCE_GROUP="rg-ml-algorithms-${ENVIRONMENT}"
ACR_NAME="crmlalgo${ENVIRONMENT}"

az acr login --name "${ACR_NAME}"
ACR_LOGIN_SERVER=$(az acr show --name "${ACR_NAME}" --query loginServer --output tsv)

docker build -t "${ACR_LOGIN_SERVER}/ml-algorithms:${ALGORITHM_NAME}-latest" \
    -f mlops/common/Dockerfile \
    "${REPO_ROOT}"

docker push "${ACR_LOGIN_SERVER}/ml-algorithms:${ALGORITHM_NAME}-latest"

# --- Step 4: Register model in Azure ML ---
echo "=== Registering model ==="
WORKSPACE="mlw-algorithms-${ENVIRONMENT}"

az ml model create \
    --name "${ALGORITHM_NAME}-model" \
    --path "${MODEL_DIR}" \
    --resource-group "${RESOURCE_GROUP}" \
    --workspace-name "${WORKSPACE}" \
    --type custom_model

# --- Step 5: Deploy with Terraform ---
echo "=== Deploying infrastructure with Terraform ==="
cd "${REPO_ROOT}/mlops/azure_ml/terraform"

terraform init -backend-config="key=azureml/${ALGORITHM_NAME}/${ENVIRONMENT}/terraform.tfstate"

terraform apply \
    -var="algorithm_name=${ALGORITHM_NAME}" \
    -var="environment=${ENVIRONMENT}" \
    -var="location=${LOCATION}" \
    -auto-approve

# --- Step 6: Test endpoint ---
echo "=== Testing endpoint ==="
ENDPOINT_NAME="${ALGORITHM_NAME}-${ENVIRONMENT}"

SCORING_URI=$(az ml online-endpoint show \
    --name "${ENDPOINT_NAME}" \
    --resource-group "${RESOURCE_GROUP}" \
    --workspace-name "${WORKSPACE}" \
    --query scoring_uri --output tsv)

ENDPOINT_KEY=$(az ml online-endpoint get-credentials \
    --name "${ENDPOINT_NAME}" \
    --resource-group "${RESOURCE_GROUP}" \
    --workspace-name "${WORKSPACE}" \
    --query primaryKey --output tsv)

curl -s -X POST "${SCORING_URI}" \
    -H "Authorization: Bearer ${ENDPOINT_KEY}" \
    -H "Content-Type: application/json" \
    -d '{"data": [[1.0, 2.0, 3.0, 4.0, 5.0]]}'

echo ""
echo "=== Deployment complete: ${ENDPOINT_NAME} ==="

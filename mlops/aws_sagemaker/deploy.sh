#!/usr/bin/env bash
#
# Deploy any algorithm to AWS SageMaker.
#
# Usage:
#   ./deploy.sh linear_regression_sklearn dev
#   ./deploy.sh random_forest_sklearn prod
#   ./deploy.sh arima_sklearn staging
#
# To deploy a DIFFERENT algorithm, just change the first argument.
# Every algorithm in this repo follows the same interface.

set -euo pipefail

ALGORITHM_NAME="${1:?Usage: ./deploy.sh <algorithm_name> <environment>}"
ENVIRONMENT="${2:-dev}"
AWS_REGION="${AWS_REGION:-us-east-1}"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

echo "=== Deploying ${ALGORITHM_NAME} to SageMaker (${ENVIRONMENT}) ==="

# --- Step 1: Determine algorithm path ---
# Map algorithm name to its directory
find_algorithm() {
    local name="$1"
    local found
    found=$(find "${REPO_ROOT}" -name "${name}.py" -not -path "*/mlops/*" -not -path "*/.claude/*" | head -1)
    if [[ -z "$found" ]]; then
        echo "ERROR: Algorithm '${name}' not found in repository" >&2
        exit 1
    fi
    # Return path relative to repo root
    echo "${found#${REPO_ROOT}/}"
}

ALGORITHM_PATH=$(find_algorithm "${ALGORITHM_NAME}")
echo "Found algorithm at: ${ALGORITHM_PATH}"

# --- Step 2: Train locally and package model ---
echo "=== Training model locally ==="
MODEL_DIR="/tmp/ml-models/${ALGORITHM_NAME}"
mkdir -p "${MODEL_DIR}"

cd "${REPO_ROOT}"
python mlops/common/train_wrapper.py \
    --algorithm-path "${ALGORITHM_PATH}" \
    --output-dir "${MODEL_DIR}"

# Package model as tar.gz (SageMaker requirement)
echo "=== Packaging model artifacts ==="
cd "${MODEL_DIR}"
tar -czf model.tar.gz model.pkl metadata.json
cd "${REPO_ROOT}"

# --- Step 3: Upload model to S3 ---
echo "=== Uploading model to S3 ==="
BUCKET="ml-algorithms-models-${ENVIRONMENT}"
aws s3 cp "${MODEL_DIR}/model.tar.gz" \
    "s3://${BUCKET}/${ALGORITHM_NAME}/model.tar.gz" \
    --region "${AWS_REGION}"

# --- Step 4: Build and push Docker container ---
echo "=== Building and pushing container ==="
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REPO="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/ml-algorithms-${ENVIRONMENT}"

# Login to ECR
aws ecr get-login-password --region "${AWS_REGION}" | \
    docker login --username AWS --password-stdin "${ECR_REPO}"

# Build container
docker build -t "ml-algorithms:${ALGORITHM_NAME}-latest" \
    -f mlops/common/Dockerfile \
    "${REPO_ROOT}"

# Tag and push
docker tag "ml-algorithms:${ALGORITHM_NAME}-latest" \
    "${ECR_REPO}:${ALGORITHM_NAME}-latest"
docker push "${ECR_REPO}:${ALGORITHM_NAME}-latest"

# --- Step 5: Deploy with Terraform ---
echo "=== Deploying infrastructure with Terraform ==="
cd "${REPO_ROOT}/mlops/aws_sagemaker/terraform"

terraform init -backend-config="key=sagemaker/${ALGORITHM_NAME}/${ENVIRONMENT}/terraform.tfstate"

terraform apply \
    -var="algorithm_name=${ALGORITHM_NAME}" \
    -var="environment=${ENVIRONMENT}" \
    -var="aws_region=${AWS_REGION}" \
    -auto-approve

# --- Step 6: Verify deployment ---
echo "=== Verifying endpoint ==="
ENDPOINT_NAME="${ALGORITHM_NAME}-${ENVIRONMENT}"

# Wait for endpoint to be InService
aws sagemaker wait endpoint-in-service \
    --endpoint-name "${ENDPOINT_NAME}" \
    --region "${AWS_REGION}"

echo "=== Testing endpoint ==="
aws sagemaker-runtime invoke-endpoint \
    --endpoint-name "${ENDPOINT_NAME}" \
    --content-type "application/json" \
    --body '{"instances": [[1.0, 2.0, 3.0, 4.0, 5.0]]}' \
    --region "${AWS_REGION}" \
    /tmp/sagemaker-response.json

echo "Response:"
cat /tmp/sagemaker-response.json
echo ""
echo "=== Deployment complete: ${ENDPOINT_NAME} ==="

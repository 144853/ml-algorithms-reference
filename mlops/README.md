# MLOps Deployment Guide

Deploy any algorithm from this repository to **AWS SageMaker**, **Azure ML**, or **Google Vertex AI** using Terraform and CI/CD pipelines.

## Key Design Principle

All 105 algorithms in this repository follow an identical interface (`generate_data()`, `train()`, `validate()`, `test()`). The MLOps scripts are **algorithm-agnostic** — to deploy any algorithm, you only change the algorithm name parameter. No code modifications needed.

```bash
# Deploy linear regression
./deploy.sh linear_regression_sklearn dev

# Deploy random forest — same script, different name
./deploy.sh random_forest_sklearn dev

# Deploy ANY algorithm — just change the name
./deploy.sh xgboost_sklearn prod
./deploy.sh lstm_pytorch staging
./deploy.sh kmeans_sklearn dev
```

## Folder Structure

```
mlops/
├── README.md                          # This file
├── common/                            # Shared components (all platforms)
│   ├── Dockerfile                     # Universal container image
│   ├── serve.py                       # REST API server (SageMaker + Azure + Vertex AI endpoints)
│   ├── train_wrapper.py               # Generic training script for any algorithm
│   └── requirements-serve.txt         # Serving dependencies
│
├── aws_sagemaker/                     # AWS SageMaker deployment
│   ├── deploy.sh                      # One-command deploy script
│   └── terraform/
│       ├── main.tf                    # Provider & backend config
│       ├── variables.tf               # Configurable parameters
│       ├── sagemaker.tf               # S3, ECR, IAM, SageMaker Model/Endpoint/Autoscaling
│       ├── outputs.tf                 # Endpoint name, ARN, bucket, ECR URL
│       └── terraform.tfvars.example   # Example variable values
│
├── azure_ml/                          # Azure Machine Learning deployment
│   ├── deploy.sh                      # One-command deploy script
│   └── terraform/
│       ├── main.tf                    # Provider & backend config
│       ├── variables.tf               # Configurable parameters
│       ├── azureml.tf                 # Resource Group, Storage, ACR, Workspace, Compute, Endpoint
│       ├── outputs.tf                 # Workspace, endpoint, ACR details
│       └── terraform.tfvars.example   # Example variable values
│
├── google_ml/                         # Google Cloud Vertex AI deployment
│   ├── deploy.sh                      # One-command deploy script
│   └── terraform/
│       ├── main.tf                    # Provider & backend config
│       ├── variables.tf               # Configurable parameters
│       ├── vertex_ai.tf               # GCS, Artifact Registry, IAM, Vertex AI Model/Endpoint/Monitoring
│       ├── outputs.tf                 # Endpoint ID, model ID, bucket, AR path
│       └── terraform.tfvars.example   # Example variable values
│
└── ci_cd/                             # CI/CD pipelines
    ├── github_actions/deploy.yml      # GitHub Actions workflow
    ├── gitlab/.gitlab-ci.yml          # GitLab CI pipeline
    └── jenkins/Jenkinsfile            # Jenkins pipeline
```

---

## Quick Start

### Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Terraform | >= 1.5.0 | Infrastructure provisioning |
| Docker | >= 24.0 | Container builds |
| Python | >= 3.11 | Training wrapper |
| AWS CLI / Azure CLI / gcloud | Latest | Cloud provider CLI |

### 1. AWS SageMaker

```bash
# Configure AWS credentials
aws configure

# Deploy (example: linear regression to dev)
cd mlops/aws_sagemaker
./deploy.sh linear_regression_sklearn dev

# Deploy a different algorithm (same command, different name)
./deploy.sh random_forest_sklearn prod
```

**What gets created:**
- S3 bucket (versioned, encrypted) for model artifacts
- ECR repository with lifecycle policies
- IAM role with least-privilege SageMaker + S3 + ECR access
- SageMaker Model, Endpoint Configuration, and Endpoint
- Data capture enabled for model monitoring
- Optional autoscaling (target: 1000 invocations/instance)
- CloudWatch log group (30-day retention)

**Terraform only (without deploy.sh):**
```bash
cd mlops/aws_sagemaker/terraform
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars — change algorithm_name to your target

terraform init
terraform plan
terraform apply
```

### 2. Azure Machine Learning

```bash
# Login to Azure
az login

# Deploy
cd mlops/azure_ml
./deploy.sh linear_regression_sklearn dev

# Different algorithm
./deploy.sh random_forest_sklearn prod
```

**What gets created:**
- Resource Group
- Storage Account (versioned blobs)
- Key Vault
- Application Insights
- Container Registry (ACR)
- Azure ML Workspace
- Compute Cluster for training (auto-scales 0-4 nodes)
- Managed Online Endpoint with key auth
- Optional autoscaling (request-rate based)

### 3. Google Cloud Vertex AI

```bash
# Login to GCP
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Deploy
cd mlops/google_ml
./deploy.sh linear_regression_sklearn dev YOUR_PROJECT_ID

# Different algorithm
./deploy.sh random_forest_sklearn prod YOUR_PROJECT_ID
```

**What gets created:**
- Required APIs enabled automatically
- GCS bucket (versioned) for model artifacts
- Artifact Registry for Docker images
- Service account with Vertex AI + Storage + AR permissions
- Vertex AI Model with custom container
- Vertex AI Endpoint with model deployment
- Cloud Monitoring alert policy for error rate

---

## CI/CD Pipelines

### GitHub Actions

The workflow at `ci_cd/github_actions/deploy.yml` supports:

- **Manual dispatch**: Choose algorithm, environment, and cloud provider from the UI
- **Auto-deploy on push**: Triggers on changes to algorithm files, deploys to dev

**Required secrets:**

| Secret | AWS | Azure | GCP |
|--------|-----|-------|-----|
| Role/Identity | `AWS_ROLE_ARN` | `AZURE_CLIENT_ID` | `GCP_WORKLOAD_IDENTITY_PROVIDER` |
| Account | `AWS_ACCOUNT_ID` | `AZURE_TENANT_ID` | `GCP_PROJECT_ID` |
| Region | `AWS_REGION` | — | `GCP_REGION` |
| Additional | — | `AZURE_SUBSCRIPTION_ID` | `GCP_SERVICE_ACCOUNT` |

**Trigger manually:**
```bash
gh workflow run deploy.yml \
  -f algorithm_name=random_forest_sklearn \
  -f environment=staging \
  -f cloud_provider=aws_sagemaker
```

### GitLab CI

Copy `ci_cd/gitlab/.gitlab-ci.yml` to your repo root. Set pipeline variables:
- `ALGORITHM_NAME`, `ENVIRONMENT`, `CLOUD_PROVIDER`
- Cloud-specific credentials as CI/CD variables

### Jenkins

Use `ci_cd/jenkins/Jenkinsfile`. Configure:
- Pipeline parameters select algorithm, environment, and provider
- Credential bindings for AWS/Azure/GCP

---

## How It Works: Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    CI/CD Pipeline                         │
│  (GitHub Actions / GitLab CI / Jenkins)                  │
├─────────────┬──────────────┬────────────────────────────┤
│  1. Validate │  2. Train     │  3. Build Container        │
│  Find algo   │  train_wrap.. │  Docker build              │
│  file in repo│  → model.pkl  │  → push to registry        │
├─────────────┴──────────────┴────────────────────────────┤
│                                                           │
│  4. Terraform Apply (infrastructure as code)             │
│                                                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │ AWS SageMaker│  │   Azure ML   │  │ Vertex AI    │   │
│  │              │  │              │  │              │   │
│  │ S3 Bucket    │  │ Storage Acct │  │ GCS Bucket   │   │
│  │ ECR Registry │  │ ACR Registry │  │ Artifact Reg │   │
│  │ IAM Role     │  │ ML Workspace │  │ Service Acct │   │
│  │ SM Model     │  │ Compute Clst │  │ VA Model     │   │
│  │ SM Endpoint  │  │ Online Endpt │  │ VA Endpoint  │   │
│  │ Auto-scaling │  │ Auto-scaling │  │ Monitoring   │   │
│  └──────────────┘  └──────────────┘  └──────────────┘   │
│                                                           │
│  5. Verify: test inference endpoint                      │
└─────────────────────────────────────────────────────────┘
```

### The Generic Serving Layer

`common/serve.py` implements endpoints for all three platforms in one container:

| Platform | Health Check | Inference |
|----------|-------------|-----------|
| SageMaker | `GET /ping` | `POST /invocations` |
| Azure ML | `GET /health` | `POST /score` |
| Vertex AI | `GET /health` | `POST /predict` |

### The Generic Training Wrapper

`common/train_wrapper.py` dynamically imports any algorithm and runs its standard pipeline:

```python
# This works for ALL 105 algorithms because they share the same interface
python train_wrapper.py --algorithm-path 01_regression/linear_regression_sklearn.py
python train_wrapper.py --algorithm-path 02_classification/random_forest_sklearn.py
python train_wrapper.py --algorithm-path 06_computer_vision/cnn_pytorch.py
```

---

## Deploying Different Algorithms

### Switching Algorithms

Every deploy script and Terraform config accepts `algorithm_name` as a parameter. The full list of deployable algorithms:

**Regression (01_regression/):**
`linear_regression_sklearn`, `linear_regression_numpy`, `linear_regression_pytorch`,
`polynomial_regression_sklearn`, `ridge_regression_sklearn`, `lasso_regression_sklearn`,
`elasticnet_sklearn`, `xgboost_sklearn`... and all variants

**Classification (02_classification/):**
`logistic_regression_sklearn`, `random_forest_sklearn`, `decision_tree_sklearn`,
`svm_sklearn`, `knn_sklearn`, `naive_bayes_sklearn`, `gradient_boosting_sklearn`,
`xgboost_sklearn`, `lightgbm_sklearn`, `catboost_sklearn`... and all variants

**Clustering (03_clustering/):**
`kmeans_sklearn`, `dbscan_sklearn`, `hierarchical_clustering_sklearn`... and all variants

**Time Series (04_time_series/):**
`arima_sklearn`, `prophet_sklearn`, `lstm_pytorch`... and all variants

**NLP (05_nlp/):**
`sentiment_analysis_sklearn`, `text_classification_sklearn`... and all variants

**Computer Vision (06_computer_vision/):**
`cnn_pytorch`, `transfer_learning_pytorch`... and all variants

**And more:** recommendation, anomaly detection, dimensionality reduction categories.

### Example: Deploy 3 Different Algorithms

```bash
# Deploy regression model to AWS
./mlops/aws_sagemaker/deploy.sh linear_regression_sklearn prod

# Deploy classifier to Azure
./mlops/azure_ml/deploy.sh random_forest_sklearn prod

# Deploy time series to GCP
./mlops/google_ml/deploy.sh arima_sklearn prod my-project
```

---

## Environment Management

| Environment | Purpose | Instance Size | Autoscaling |
|-------------|---------|---------------|-------------|
| `dev` | Development/testing | Small (ml.m5.large) | Off |
| `staging` | Pre-production validation | Medium (ml.m5.xlarge) | Optional |
| `prod` | Production serving | Large (ml.m5.2xlarge) | On |

Each environment gets isolated infrastructure (separate S3 buckets, endpoints, IAM roles, etc.) via the `environment` variable.

---

## Terraform State Management

| Provider | Backend | State Path |
|----------|---------|-----------|
| AWS | S3 + DynamoDB | `sagemaker/{algorithm}/{env}/terraform.tfstate` |
| Azure | Azure Blob Storage | `azureml/{algorithm}/{env}/terraform.tfstate` |
| GCP | GCS | `vertex-ai/{algorithm}/{env}/terraform.tfstate` |

Each algorithm + environment combination gets its own state file, so deployments are fully independent.

---

## Cleanup

```bash
# AWS
cd mlops/aws_sagemaker/terraform
terraform destroy -var="algorithm_name=linear_regression_sklearn" -var="environment=dev"

# Azure
cd mlops/azure_ml/terraform
terraform destroy -var="algorithm_name=linear_regression_sklearn" -var="environment=dev"

# GCP
cd mlops/google_ml/terraform
terraform destroy -var="project_id=my-project" -var="algorithm_name=linear_regression_sklearn" -var="environment=dev"
```

---

## Cost Considerations

| Resource | AWS (dev) | Azure (dev) | GCP (dev) |
|----------|-----------|-------------|-----------|
| Endpoint (ml.m5.large / DS3_v2 / n1-standard-4) | ~$0.115/hr | ~$0.19/hr | ~$0.19/hr |
| Storage (model artifacts) | ~$0.023/GB/mo | ~$0.018/GB/mo | ~$0.020/GB/mo |
| Container Registry | ~$0.10/GB/mo | ~$0.10/GB/mo | ~$0.10/GB/mo |

**Tip:** Use `terraform destroy` to tear down dev/staging endpoints when not in use. Training compute (Azure ML Compute Cluster) auto-scales to 0 nodes after 5 minutes of idle.

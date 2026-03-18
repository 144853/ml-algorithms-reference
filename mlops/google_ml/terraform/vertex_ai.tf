# --- Enable Required APIs ---

resource "google_project_service" "apis" {
  for_each = toset([
    "aiplatform.googleapis.com",
    "artifactregistry.googleapis.com",
    "cloudbuild.googleapis.com",
    "storage.googleapis.com",
    "iam.googleapis.com",
  ])

  project = var.project_id
  service = each.value

  disable_dependent_services = false
  disable_on_destroy         = false
}

# --- GCS Bucket for Model Artifacts ---

resource "google_storage_bucket" "model_artifacts" {
  name     = "${var.project_id}-ml-models-${var.environment}"
  location = var.region

  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }

  labels = {
    project     = "ml-algorithms-reference"
    environment = var.environment
    managed-by  = "terraform"
  }

  depends_on = [google_project_service.apis]
}

# --- Artifact Registry for Docker Images ---

resource "google_artifact_registry_repository" "ml_container" {
  location      = var.region
  repository_id = "ml-algorithms-${var.environment}"
  format        = "DOCKER"
  description   = "ML algorithm containers for ${var.environment}"

  labels = {
    environment = var.environment
  }

  depends_on = [google_project_service.apis]
}

# --- Service Account for Vertex AI ---

resource "google_service_account" "vertex_ai" {
  account_id   = "vertex-ai-${var.environment}"
  display_name = "Vertex AI Service Account (${var.environment})"
}

resource "google_project_iam_member" "vertex_ai_user" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.vertex_ai.email}"
}

resource "google_project_iam_member" "storage_admin" {
  project = var.project_id
  role    = "roles/storage.objectAdmin"
  member  = "serviceAccount:${google_service_account.vertex_ai.email}"
}

resource "google_project_iam_member" "artifact_registry_reader" {
  project = var.project_id
  role    = "roles/artifactregistry.reader"
  member  = "serviceAccount:${google_service_account.vertex_ai.email}"
}

# --- Vertex AI Model ---

resource "google_vertex_ai_model" "algorithm" {
  display_name = "${var.algorithm_name}-${var.environment}"
  region       = var.region

  container_spec {
    image_uri = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.ml_container.repository_id}/ml-algorithms:${var.algorithm_name}-latest"

    ports {
      container_port = 8080
    }

    env {
      name  = "MODEL_DIR"
      value = "/opt/ml/model"
    }

    predict_route = "/predict"
    health_route  = "/health"
  }

  artifact_uri = "gs://${google_storage_bucket.model_artifacts.name}/${var.algorithm_name}/"

  labels = {
    algorithm   = replace(var.algorithm_name, "_", "-")
    environment = var.environment
  }

  depends_on = [google_project_service.apis]
}

# --- Vertex AI Endpoint ---

resource "google_vertex_ai_endpoint" "algorithm" {
  display_name = "${var.algorithm_name}-${var.environment}"
  region       = var.region
  description  = "Endpoint for ${var.algorithm_name} (${var.environment})"

  labels = {
    algorithm   = replace(var.algorithm_name, "_", "-")
    environment = var.environment
  }

  depends_on = [google_project_service.apis]
}

# --- Deploy Model to Endpoint ---
# Vertex AI model deployment is handled via gcloud CLI
# because the Terraform provider doesn't support deployModel natively.

resource "null_resource" "deploy_model" {
  triggers = {
    algorithm_name = var.algorithm_name
    environment    = var.environment
    endpoint_id    = google_vertex_ai_endpoint.algorithm.name
    model_id       = google_vertex_ai_model.algorithm.name
  }

  provisioner "local-exec" {
    command = <<-EOT
      gcloud ai endpoints deploy-model \
        "${google_vertex_ai_endpoint.algorithm.name}" \
        --project="${var.project_id}" \
        --region="${var.region}" \
        --model="${google_vertex_ai_model.algorithm.name}" \
        --display-name="${var.algorithm_name}-deployment" \
        --machine-type="${var.machine_type}" \
        --min-replica-count=${var.min_replica_count} \
        --max-replica-count=${var.enable_autoscaling ? var.max_replica_count : var.min_replica_count} \
        --service-account="${google_service_account.vertex_ai.email}" \
        --traffic-split="0=100"
    EOT
  }

  depends_on = [
    google_vertex_ai_endpoint.algorithm,
    google_vertex_ai_model.algorithm,
  ]
}

# --- Monitoring ---

resource "google_monitoring_alert_policy" "endpoint_errors" {
  display_name = "Vertex AI ${var.algorithm_name} Error Rate (${var.environment})"
  combiner     = "OR"

  conditions {
    display_name = "High error rate"

    condition_threshold {
      filter          = "resource.type = \"aiplatform.googleapis.com/Endpoint\" AND metric.type = \"aiplatform.googleapis.com/prediction/online/error_count\""
      comparison      = "COMPARISON_GT"
      threshold_value = 10
      duration        = "300s"

      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_RATE"
      }
    }
  }

  alert_strategy {
    auto_close = "1800s"
  }
}

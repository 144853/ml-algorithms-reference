output "endpoint_id" {
  description = "Vertex AI endpoint ID"
  value       = google_vertex_ai_endpoint.algorithm.name
}

output "endpoint_display_name" {
  description = "Vertex AI endpoint display name"
  value       = google_vertex_ai_endpoint.algorithm.display_name
}

output "model_id" {
  description = "Vertex AI model ID"
  value       = google_vertex_ai_model.algorithm.name
}

output "model_artifact_bucket" {
  description = "GCS bucket for model artifacts"
  value       = google_storage_bucket.model_artifacts.name
}

output "artifact_registry" {
  description = "Artifact Registry repository path"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.ml_container.repository_id}"
}

output "service_account_email" {
  description = "Vertex AI service account email"
  value       = google_service_account.vertex_ai.email
}

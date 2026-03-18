variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region for resources"
  type        = string
  default     = "us-central1"
}

variable "environment" {
  description = "Deployment environment (dev, staging, prod)"
  type        = string
  default     = "dev"

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be dev, staging, or prod."
  }
}

variable "algorithm_name" {
  description = "Name of the algorithm to deploy (e.g., linear_regression_sklearn, random_forest_sklearn)"
  type        = string
}

variable "machine_type" {
  description = "Machine type for the Vertex AI endpoint"
  type        = string
  default     = "n1-standard-4"
}

variable "min_replica_count" {
  description = "Minimum number of replicas"
  type        = number
  default     = 1
}

variable "max_replica_count" {
  description = "Maximum number of replicas"
  type        = number
  default     = 4
}

variable "enable_autoscaling" {
  description = "Enable auto-scaling for the endpoint"
  type        = bool
  default     = false
}

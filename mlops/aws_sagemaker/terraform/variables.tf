variable "aws_region" {
  description = "AWS region for SageMaker resources"
  type        = string
  default     = "us-east-1"
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

variable "instance_type" {
  description = "SageMaker instance type for the endpoint"
  type        = string
  default     = "ml.m5.large"
}

variable "instance_count" {
  description = "Number of instances for the endpoint"
  type        = number
  default     = 1
}

variable "training_instance_type" {
  description = "SageMaker instance type for training"
  type        = string
  default     = "ml.m5.xlarge"
}

variable "ecr_repository_name" {
  description = "ECR repository name for the ML container"
  type        = string
  default     = "ml-algorithms"
}

variable "model_data_bucket" {
  description = "S3 bucket for model artifacts"
  type        = string
  default     = "ml-algorithms-models"
}

variable "enable_autoscaling" {
  description = "Enable auto-scaling for the SageMaker endpoint"
  type        = bool
  default     = false
}

variable "min_capacity" {
  description = "Minimum number of instances for autoscaling"
  type        = number
  default     = 1
}

variable "max_capacity" {
  description = "Maximum number of instances for autoscaling"
  type        = number
  default     = 4
}

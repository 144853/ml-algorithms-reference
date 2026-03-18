variable "location" {
  description = "Azure region for resources"
  type        = string
  default     = "eastus"
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

variable "vm_size" {
  description = "Azure VM size for the managed endpoint"
  type        = string
  default     = "Standard_DS3_v2"
}

variable "instance_count" {
  description = "Number of instances for the endpoint"
  type        = number
  default     = 1
}

variable "training_vm_size" {
  description = "Azure VM size for training compute"
  type        = string
  default     = "Standard_DS3_v2"
}

variable "enable_autoscaling" {
  description = "Enable auto-scaling for the endpoint"
  type        = bool
  default     = false
}

variable "min_instances" {
  description = "Minimum instances for autoscaling"
  type        = number
  default     = 1
}

variable "max_instances" {
  description = "Maximum instances for autoscaling"
  type        = number
  default     = 4
}

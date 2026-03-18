output "resource_group_name" {
  description = "Resource group name"
  value       = azurerm_resource_group.ml.name
}

output "workspace_name" {
  description = "Azure ML workspace name"
  value       = azurerm_machine_learning_workspace.ml.name
}

output "workspace_id" {
  description = "Azure ML workspace ID"
  value       = azurerm_machine_learning_workspace.ml.id
}

output "endpoint_name" {
  description = "Managed online endpoint name"
  value       = "${var.algorithm_name}-${var.environment}"
}

output "container_registry" {
  description = "Container registry login server"
  value       = azurerm_container_registry.ml.login_server
}

output "storage_account_name" {
  description = "Storage account for model artifacts"
  value       = azurerm_storage_account.ml.name
}

output "application_insights_key" {
  description = "Application Insights instrumentation key"
  value       = azurerm_application_insights.ml.instrumentation_key
  sensitive   = true
}

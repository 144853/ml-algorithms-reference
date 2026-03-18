# --- Resource Group ---

resource "azurerm_resource_group" "ml" {
  name     = "rg-ml-algorithms-${var.environment}"
  location = var.location

  tags = {
    Project     = "ml-algorithms-reference"
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}

# --- Storage Account ---

resource "azurerm_storage_account" "ml" {
  name                     = "mlalgo${var.environment}${substr(md5(var.algorithm_name), 0, 6)}"
  resource_group_name      = azurerm_resource_group.ml.name
  location                 = azurerm_resource_group.ml.location
  account_tier             = "Standard"
  account_replication_type = "LRS"
  min_tls_version          = "TLS1_2"

  blob_properties {
    versioning_enabled = true
  }
}

resource "azurerm_storage_container" "models" {
  name                  = "models"
  storage_account_id    = azurerm_storage_account.ml.id
  container_access_type = "private"
}

# --- Key Vault ---

resource "azurerm_key_vault" "ml" {
  name                = "kv-ml-${var.environment}-${substr(md5(var.algorithm_name), 0, 6)}"
  location            = azurerm_resource_group.ml.location
  resource_group_name = azurerm_resource_group.ml.name
  tenant_id           = data.azurerm_client_config.current.tenant_id
  sku_name            = "standard"

  purge_protection_enabled = false
}

data "azurerm_client_config" "current" {}

# --- Application Insights ---

resource "azurerm_application_insights" "ml" {
  name                = "ai-ml-algorithms-${var.environment}"
  location            = azurerm_resource_group.ml.location
  resource_group_name = azurerm_resource_group.ml.name
  application_type    = "web"
}

# --- Container Registry ---

resource "azurerm_container_registry" "ml" {
  name                = "crmlalgo${var.environment}"
  resource_group_name = azurerm_resource_group.ml.name
  location            = azurerm_resource_group.ml.location
  sku                 = "Standard"
  admin_enabled       = true
}

# --- Azure Machine Learning Workspace ---

resource "azurerm_machine_learning_workspace" "ml" {
  name                          = "mlw-algorithms-${var.environment}"
  location                      = azurerm_resource_group.ml.location
  resource_group_name           = azurerm_resource_group.ml.name
  application_insights_id       = azurerm_application_insights.ml.id
  key_vault_id                  = azurerm_key_vault.ml.id
  storage_account_id            = azurerm_storage_account.ml.id
  container_registry_id         = azurerm_container_registry.ml.id
  public_network_access_enabled = true

  identity {
    type = "SystemAssigned"
  }
}

# --- Compute Cluster for Training ---

resource "azurerm_machine_learning_compute_cluster" "training" {
  name                          = "train-cluster"
  location                      = azurerm_resource_group.ml.location
  machine_learning_workspace_id = azurerm_machine_learning_workspace.ml.id
  vm_priority                   = "Dedicated"
  vm_size                       = var.training_vm_size

  scale_settings {
    min_node_count                       = 0
    max_node_count                       = 4
    scale_down_nodes_after_idle_duration = "PT5M"
  }

  identity {
    type = "SystemAssigned"
  }
}

# --- Managed Online Endpoint ---
# Azure ML managed endpoints are created via Azure CLI / SDK
# because the azurerm provider doesn't have native support yet.
# The null_resource below handles endpoint creation.

resource "null_resource" "managed_endpoint" {
  triggers = {
    algorithm_name = var.algorithm_name
    environment    = var.environment
    workspace      = azurerm_machine_learning_workspace.ml.name
    resource_group = azurerm_resource_group.ml.name
  }

  provisioner "local-exec" {
    command = <<-EOT
      az ml online-endpoint create \
        --name "${var.algorithm_name}-${var.environment}" \
        --resource-group "${azurerm_resource_group.ml.name}" \
        --workspace-name "${azurerm_machine_learning_workspace.ml.name}" \
        --auth-mode key
    EOT
  }

  provisioner "local-exec" {
    when    = destroy
    command = <<-EOT
      az ml online-endpoint delete \
        --name "${self.triggers.algorithm_name}-${self.triggers.environment}" \
        --resource-group "${self.triggers.resource_group}" \
        --workspace-name "${self.triggers.workspace}" \
        --yes --no-wait
    EOT
  }

  depends_on = [azurerm_machine_learning_workspace.ml]
}

# --- Managed Online Deployment ---

resource "null_resource" "managed_deployment" {
  triggers = {
    algorithm_name = var.algorithm_name
    environment    = var.environment
    instance_count = var.instance_count
  }

  provisioner "local-exec" {
    command = <<-EOT
      az ml online-deployment create \
        --name "primary" \
        --endpoint-name "${var.algorithm_name}-${var.environment}" \
        --resource-group "${azurerm_resource_group.ml.name}" \
        --workspace-name "${azurerm_machine_learning_workspace.ml.name}" \
        --model "azureml:${var.algorithm_name}-model:1" \
        --instance-type "${var.vm_size}" \
        --instance-count ${var.instance_count} \
        --all-traffic
    EOT
  }

  depends_on = [null_resource.managed_endpoint]
}

# --- Autoscaling (optional) ---

resource "azurerm_monitor_autoscale_setting" "endpoint" {
  count = var.enable_autoscaling ? 1 : 0

  name                = "autoscale-${var.algorithm_name}-${var.environment}"
  resource_group_name = azurerm_resource_group.ml.name
  location            = azurerm_resource_group.ml.location
  target_resource_id  = "${azurerm_machine_learning_workspace.ml.id}/onlineEndpoints/${var.algorithm_name}-${var.environment}/deployments/primary"

  profile {
    name = "default"

    capacity {
      default = var.instance_count
      minimum = var.min_instances
      maximum = var.max_instances
    }

    rule {
      metric_trigger {
        metric_name        = "RequestsPerMinute"
        metric_resource_id = "${azurerm_machine_learning_workspace.ml.id}/onlineEndpoints/${var.algorithm_name}-${var.environment}/deployments/primary"
        time_grain         = "PT1M"
        statistic          = "Average"
        time_window        = "PT5M"
        time_aggregation   = "Average"
        operator           = "GreaterThan"
        threshold          = 100
      }
      scale_action {
        direction = "Increase"
        type      = "ChangeCount"
        value     = "1"
        cooldown  = "PT5M"
      }
    }

    rule {
      metric_trigger {
        metric_name        = "RequestsPerMinute"
        metric_resource_id = "${azurerm_machine_learning_workspace.ml.id}/onlineEndpoints/${var.algorithm_name}-${var.environment}/deployments/primary"
        time_grain         = "PT1M"
        statistic          = "Average"
        time_window        = "PT10M"
        time_aggregation   = "Average"
        operator           = "LessThan"
        threshold          = 20
      }
      scale_action {
        direction = "Decrease"
        type      = "ChangeCount"
        value     = "1"
        cooldown  = "PT10M"
      }
    }
  }
}

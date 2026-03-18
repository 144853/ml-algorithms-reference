# --- S3 Bucket for Model Artifacts ---

resource "aws_s3_bucket" "model_artifacts" {
  bucket = "${var.model_data_bucket}-${var.environment}"
}

resource "aws_s3_bucket_versioning" "model_artifacts" {
  bucket = aws_s3_bucket.model_artifacts.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "model_artifacts" {
  bucket = aws_s3_bucket.model_artifacts.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "aws:kms"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "model_artifacts" {
  bucket                  = aws_s3_bucket.model_artifacts.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# --- ECR Repository ---

resource "aws_ecr_repository" "ml_container" {
  name                 = "${var.ecr_repository_name}-${var.environment}"
  image_tag_mutability = "IMMUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = "AES256"
  }
}

resource "aws_ecr_lifecycle_policy" "ml_container" {
  repository = aws_ecr_repository.ml_container.name

  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep last 10 images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 10
      }
      action = {
        type = "expire"
      }
    }]
  })
}

# --- IAM Role for SageMaker ---

data "aws_iam_policy_document" "sagemaker_assume_role" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["sagemaker.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "sagemaker_execution" {
  name               = "sagemaker-${var.algorithm_name}-${var.environment}"
  assume_role_policy = data.aws_iam_policy_document.sagemaker_assume_role.json
}

resource "aws_iam_role_policy_attachment" "sagemaker_full_access" {
  role       = aws_iam_role.sagemaker_execution.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

data "aws_iam_policy_document" "s3_access" {
  statement {
    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:ListBucket",
    ]
    resources = [
      aws_s3_bucket.model_artifacts.arn,
      "${aws_s3_bucket.model_artifacts.arn}/*",
    ]
  }
}

resource "aws_iam_role_policy" "s3_access" {
  name   = "s3-model-access"
  role   = aws_iam_role.sagemaker_execution.id
  policy = data.aws_iam_policy_document.s3_access.json
}

data "aws_iam_policy_document" "ecr_access" {
  statement {
    actions = [
      "ecr:GetDownloadUrlForLayer",
      "ecr:BatchGetImage",
      "ecr:GetAuthorizationToken",
    ]
    resources = ["*"]
  }
}

resource "aws_iam_role_policy" "ecr_access" {
  name   = "ecr-access"
  role   = aws_iam_role.sagemaker_execution.id
  policy = data.aws_iam_policy_document.ecr_access.json
}

# --- CloudWatch Log Group ---

resource "aws_cloudwatch_log_group" "sagemaker" {
  name              = "/aws/sagemaker/endpoints/${var.algorithm_name}-${var.environment}"
  retention_in_days = 30
}

# --- SageMaker Model ---

resource "aws_sagemaker_model" "algorithm" {
  name               = "${var.algorithm_name}-${var.environment}"
  execution_role_arn = aws_iam_role.sagemaker_execution.arn

  primary_container {
    image          = "${aws_ecr_repository.ml_container.repository_url}:${var.algorithm_name}-latest"
    model_data_url = "s3://${aws_s3_bucket.model_artifacts.id}/${var.algorithm_name}/model.tar.gz"
    environment = {
      MODEL_DIR = "/opt/ml/model"
    }
  }

  depends_on = [aws_cloudwatch_log_group.sagemaker]
}

# --- SageMaker Endpoint Configuration ---

resource "aws_sagemaker_endpoint_configuration" "algorithm" {
  name = "${var.algorithm_name}-config-${var.environment}"

  production_variants {
    variant_name           = "primary"
    model_name             = aws_sagemaker_model.algorithm.name
    initial_instance_count = var.instance_count
    instance_type          = var.instance_type
    initial_variant_weight = 1.0
  }

  data_capture_config {
    enable_capture              = true
    initial_sampling_percentage = 100
    destination_s3_uri          = "s3://${aws_s3_bucket.model_artifacts.id}/data-capture/${var.algorithm_name}"

    capture_options {
      capture_mode = "Input"
    }
    capture_options {
      capture_mode = "Output"
    }
  }
}

# --- SageMaker Endpoint ---

resource "aws_sagemaker_endpoint" "algorithm" {
  name                 = "${var.algorithm_name}-${var.environment}"
  endpoint_config_name = aws_sagemaker_endpoint_configuration.algorithm.name
}

# --- Auto Scaling (optional) ---

resource "aws_appautoscaling_target" "sagemaker" {
  count = var.enable_autoscaling ? 1 : 0

  max_capacity       = var.max_capacity
  min_capacity       = var.min_capacity
  resource_id        = "endpoint/${aws_sagemaker_endpoint.algorithm.name}/variant/primary"
  scalable_dimension = "sagemaker:variant:DesiredInstanceCount"
  service_namespace  = "sagemaker"
}

resource "aws_appautoscaling_policy" "sagemaker" {
  count = var.enable_autoscaling ? 1 : 0

  name               = "${var.algorithm_name}-scaling-${var.environment}"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.sagemaker[0].resource_id
  scalable_dimension = aws_appautoscaling_target.sagemaker[0].scalable_dimension
  service_namespace  = aws_appautoscaling_target.sagemaker[0].service_namespace

  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "SageMakerVariantInvocationsPerInstance"
    }
    target_value = 1000
  }
}

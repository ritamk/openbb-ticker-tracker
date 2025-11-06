#!/bin/bash

# Script runner for trading_llm project
# Usage: ./scripts.sh [script_name]
# Example: ./scripts.sh run_main

# Run trading_llm main module with pretty output
run_main() {
    python -m trading_llm.main --pretty
}

# Build docker image
build_docker_image() {
    docker build -t ritamk/brok .
}

# Build fresh docker image with no cache
build_docker_image_no_cache() {
    docker build --no-cache -t ritamk/brok .
}

# Run docker image
run_docker_image() {
    docker run --rm -p 8080:8080 \
        --env-file .env \
        -e PORT=8080 \
        ritamk/brok
}

# Deploy to Google Cloud Run
deploy_cloud_run() {
    echo "Deploying to Google Cloud Run..."
    
    # Check if gcloud is installed
    if ! command -v gcloud &> /dev/null; then
        echo "Error: gcloud CLI not found. Please install it first."
        echo "Visit: https://cloud.google.com/sdk/docs/install"
        exit 1
    fi
    
    # Get project ID
    PROJECT_ID="brok-c367d"
    if [ -z "$PROJECT_ID" ]; then
        echo "Error: No GCP project configured. Run: gcloud config set project PROJECT_ID"
        exit 1
    fi
    
    SERVICE_NAME="brok-api"
    REGION="asia-south2"
    IMAGE_NAME="gcr.io/$PROJECT_ID/$SERVICE_NAME"
    
    echo "Project: $PROJECT_ID"
    echo "Service: $SERVICE_NAME"
    echo "Region: $REGION"
    echo ""
    
    # Build and push image
    echo "Building and pushing Docker image..."
    # Note: Using a unique tag to force Cloud Run to pull the new image
    TIMESTAMP=$(date +%Y%m%d-%H%M%S)
    IMAGE_WITH_TAG="$IMAGE_NAME:$TIMESTAMP"
    
    gcloud builds submit --tag "$IMAGE_WITH_TAG" || {
        echo "Error: Build failed"
        exit 1
    }
    
    # Also tag as latest
    gcloud container images add-tag "$IMAGE_WITH_TAG" "$IMAGE_NAME:latest" --quiet
    
    echo ""
    echo "Deploying to Cloud Run with image: $IMAGE_WITH_TAG"
    gcloud run deploy "$SERVICE_NAME" \
        --image "$IMAGE_WITH_TAG" \
        --platform managed \
        --region "$REGION" \
        --allow-unauthenticated \
        --memory 2Gi \
        --cpu 2 \
        --timeout 300 \
        --max-instances 10 || {
        echo "Error: Deployment failed"
        exit 1
    }
    
    echo ""
    echo "✅ Deployment complete!"
    echo ""
    echo "Service URL:"
    gcloud run services describe "$SERVICE_NAME" --region "$REGION" --format='value(status.url)'
    echo ""
    echo "To view logs:"
    echo "  gcloud run services logs read $SERVICE_NAME --region $REGION --limit 50"
}

# Display available scripts
show_help() {
    echo "Available scripts:"
    echo "  run_main  - Execute trading_llm main module with pretty output"
    echo "  build_docker_image - Build docker image"
    echo "  build_docker_image_no_cache - Build fresh docker image with no cache"
    echo "  run_docker_image - Run docker image"
    echo "  deploy_cloud_run - Deploy to Google Cloud Run"
    # Add more scripts here as you add them
}

# Main script dispatcher
main() {
    local script_name="${1:-run_main}"  # Default to run_main if no argument provided
    
    case "$script_name" in
        run_main)
            run_main
            ;;
        build_docker_image)
            build_docker_image
            ;;
        build_docker_image_no_cache)
            build_docker_image_no_cache
            ;;
        run_docker_image)
            run_docker_image
            ;;
        deploy_cloud_run)
            deploy_cloud_run
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            echo "Unknown script: $script_name"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# Run the main function with all arguments
main "$@"

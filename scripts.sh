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
        -v ~/.openbb_platform:/root/.openbb_platform:ro \
        ritamk/brok
}

# Display available scripts
show_help() {
    echo "Available scripts:"
    echo "  run_main  - Execute trading_llm main module with pretty output"
    echo "  build_docker_image - Build docker image"
    echo "  build_docker_image_no_cache - Build fresh docker image with no cache"
    echo "  run_docker_image - Run docker image"
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

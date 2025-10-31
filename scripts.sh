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
    docker build -t trading_llm .
}

# Run docker image
run_docker_image() {
    docker run --rm -p 8080:8080 \
        --env-file .env \
        -e PORT=8080 \
        trading-llm-api
}

# Display available scripts
show_help() {
    echo "Available scripts:"
    echo "  run_main  - Execute trading_llm main module with pretty output"
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

#!/bin/bash

# Script to copy training runs from mounted remote directory to local logs
# This is useful when you have the remote directory mounted via SSHFS

set -e

# Configuration
MOUNT_POINT="remote_mount"
LOCAL_LOGS="logs"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [env_type] [experiment_name] [run_name]"
    echo ""
    echo "Arguments:"
    echo "  env_type         Environment type (required)"
    echo "  experiment_name  Experiment name (required)"
    echo "  run_name         Specific run name (optional, defaults to latest)"
    echo ""
    echo "Examples:"
    echo "  $0 vanilla g1                    # Copy latest vanilla/g1 run"
    echo "  $0 custom g1                     # Copy latest custom/g1 run"
    echo "  $0 vanilla g1 2025-06-02_14-07-15  # Copy specific run"
    echo ""
    echo "Available environment types:"
    echo "  vanilla, custom, clf, ref_tracking, stair, clf_vdot,"
    echo "  height-scan-flat, flat-hzd, flat-hzd-no-dr, stair-hzd"
}

# Check if mount point exists and is accessible
check_mount() {
    if [[ ! -d "$MOUNT_POINT" ]]; then
        print_error "Mount point $MOUNT_POINT does not exist"
        print_info "Please run: ./scripts/mount_remote.sh mount"
        exit 1
    fi

    if ! mountpoint -q "$MOUNT_POINT" 2>/dev/null; then
        print_error "Mount point $MOUNT_POINT is not mounted"
        print_info "Please run: ./scripts/mount_remote.sh mount"
        exit 1
    fi
}

# List available runs
list_runs() {
    local env_type="$1"
    local experiment_name="$2"
    local remote_path="$MOUNT_POINT/logs/g1_policies/$env_type/$experiment_name"

    if [[ ! -d "$remote_path" ]]; then
        print_error "Remote path does not exist: $remote_path"
        return 1
    fi

    print_info "Available runs for $env_type/$experiment_name:"
    ls -1t "$remote_path" | head -10 | sed 's/^/  /'

    if [[ $(ls -1 "$remote_path" | wc -l) -gt 10 ]]; then
        print_info "... and $(($(ls -1 "$remote_path" | wc -l) - 10)) more runs"
    fi
}

# Main script logic
main() {
    # Check if help is requested
    if [[ "$1" == "-h" || "$1" == "--help" ]]; then
        show_usage
        exit 0
    fi

    # Check arguments
    if [[ $# -lt 2 ]]; then
        print_error "Missing required arguments"
        show_usage
        exit 1
    fi

    ENV_TYPE="$1"
    EXPERIMENT_NAME="$2"
    RUN_NAME="${3:-}"

    print_info "Copying training run for $ENV_TYPE/$EXPERIMENT_NAME"

    # Check mount
    check_mount

    # Define paths
    REMOTE_PATH="$MOUNT_POINT/logs/g1_policies/$ENV_TYPE/$EXPERIMENT_NAME"
    LOCAL_PATH="$LOCAL_LOGS/g1_policies/$ENV_TYPE/$EXPERIMENT_NAME"

    # Check if remote path exists
    if [[ ! -d "$REMOTE_PATH" ]]; then
        print_error "Remote path does not exist: $REMOTE_PATH"
        print_info "Available environments on remote:"
        ls -1 "$MOUNT_POINT/logs/g1_policies/" | sed 's/^/  /'
        exit 1
    fi

    # If no specific run name provided, find the latest
    if [[ -z "$RUN_NAME" ]]; then
        RUN_NAME=$(ls -1t "$REMOTE_PATH" | head -1)
        print_info "Using latest run: $RUN_NAME"
    fi

    # Check if the specific run exists
    if [[ ! -d "$REMOTE_PATH/$RUN_NAME" ]]; then
        print_error "Run $RUN_NAME does not exist in $REMOTE_PATH"
        list_runs "$ENV_TYPE" "$EXPERIMENT_NAME"
        exit 1
    fi

    # Create local directory
    mkdir -p "$LOCAL_PATH"

    # Copy the run
    REMOTE_RUN_PATH="$REMOTE_PATH/$RUN_NAME"
    LOCAL_RUN_PATH="$LOCAL_PATH/$RUN_NAME"

    print_info "Copying from: $REMOTE_RUN_PATH"
    print_info "Copying to: $LOCAL_RUN_PATH"

    # Use rsync for efficient copying
    if rsync -av --progress "$REMOTE_RUN_PATH/" "$LOCAL_RUN_PATH/"; then
        print_success "Successfully copied training run!"
        print_info "Local path: $LOCAL_RUN_PATH"

        # Show summary
        MODEL_COUNT=$(find "$LOCAL_RUN_PATH" -name "model_*.pt" | wc -l)
        print_info "Found $MODEL_COUNT model checkpoint(s)"

        LATEST_CHECKPOINT=$(find "$LOCAL_RUN_PATH" -name "model_*.pt" | sort -V | tail -1)
        if [[ -n "$LATEST_CHECKPOINT" ]]; then
            print_success "Latest checkpoint: $(basename "$LATEST_CHECKPOINT")"
        fi

        # Show directory size
        SIZE=$(du -sh "$LOCAL_RUN_PATH" | cut -f1)
        print_info "Directory size: $SIZE"

    else
        print_error "Failed to copy training run"
        exit 1
    fi
}

# Run main function
main "$@"

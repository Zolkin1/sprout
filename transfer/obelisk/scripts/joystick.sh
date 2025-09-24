#!/bin/bash

# Joystick Setup Script for Obelisk
# This script detects joystick devices and sets up proper permissions

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KNOWN_NAMES_FILE="$SCRIPT_DIR/resource/joystick_names.txt"

echo "=== Joystick Setup Script ==="
echo "Detecting input devices..."

# Check if evtest is available
if ! command -v evtest &> /dev/null; then
    echo "Error: evtest is not installed. Please install it first."
    echo "Run: sudo apt-get install evtest"
    exit 1
fi

# Get list of input devices and their names
echo "Scanning for input devices..."

# Use a different approach - send empty input to evtest to get device list
DEVICE_INFO=$(echo "" | timeout 5 evtest 2>&1 | grep -E "^/dev/input/event[0-9]+:" | sed 's/\t/ /g')

# If that doesn't work, try parsing /proc/bus/input/devices as fallback
if [ -z "$DEVICE_INFO" ]; then
    echo "Trying alternative device detection method..."
    # Parse /proc/bus/input/devices for event handlers and names
    DEVICE_INFO=""
    while IFS= read -r line; do
        if [[ $line =~ ^N:\ Name=\"(.+)\"$ ]]; then
            current_name="${BASH_REMATCH[1]}"
        elif [[ $line =~ ^H:\ Handlers=.*event([0-9]+) ]]; then
            event_num="${BASH_REMATCH[1]}"
            if [ ! -z "$current_name" ]; then
                DEVICE_INFO="$DEVICE_INFO"$'\n'"/dev/input/event$event_num: $current_name"
            fi
            current_name=""
        fi
    done < /proc/bus/input/devices
fi

if [ -z "$DEVICE_INFO" ]; then
    echo "No input devices found. Make sure your joystick is connected."
    exit 1
fi

echo "Found input devices:"
echo "$DEVICE_INFO"
echo ""

# Read known joystick names
if [ ! -f "$KNOWN_NAMES_FILE" ]; then
    echo "Warning: Known joystick names file not found at $KNOWN_NAMES_FILE"
    echo "Creating empty file..."
    touch "$KNOWN_NAMES_FILE"
fi

# Check for known joystick names
FOUND_DEVICE=""
FOUND_EVENT=""

while IFS= read -r known_name; do
    [ -z "$known_name" ] && continue

    # Look for this known name in the device list
    MATCH=$(echo "$DEVICE_INFO" | grep -i "$known_name")
    if [ ! -z "$MATCH" ]; then
        FOUND_DEVICE="$known_name"
        FOUND_EVENT=$(echo "$MATCH" | grep -o "/dev/input/event[0-9]\+" | head -1)
        break
    fi
done < "$KNOWN_NAMES_FILE"

if [ ! -z "$FOUND_DEVICE" ]; then
    echo "Found known joystick: $FOUND_DEVICE"
    echo "Device: $FOUND_EVENT"

    # Set permissions
    echo "Setting permissions for $FOUND_EVENT..."
    sudo chmod 666 "$FOUND_EVENT"

    if [ $? -eq 0 ]; then
        echo " Permissions set successfully!"

        # Test with ROS2 if available
        if command -v ros2 &> /dev/null; then
            echo ""
            echo "Testing ROS2 joystick detection..."
            ros2 run joy joy_enumerate_devices 2>/dev/null || echo "Note: ROS2 joy package may not be available in this environment"
        fi

        echo ""
        echo "Joystick setup completed successfully!"
        echo "Device: $FOUND_EVENT ($FOUND_DEVICE)"
    else
        echo "Error: Failed to set permissions. Make sure you have sudo access."
        exit 1
    fi
else
    echo "No known joystick devices found automatically."
    echo ""
    echo "Available devices:"

    # Show numbered list of devices
    i=1
    declare -a device_paths
    declare -a device_names

    while IFS= read -r line; do
        if [[ $line =~ ^(/dev/input/event[0-9]+):[[:space:]]*(.+)$ ]]; then
            device_path="${BASH_REMATCH[1]}"
            device_name="${BASH_REMATCH[2]}"
            device_paths[$i]="$device_path"
            device_names[$i]="$device_name"
            echo "$i) $device_path: $device_name"
            ((i++))
        fi
    done <<< "$DEVICE_INFO"

    total_devices=$((i-1))

    if [ $total_devices -eq 0 ]; then
        echo "No devices found to configure."
        exit 1
    fi

    echo ""
    echo "Please select the joystick device (1-$total_devices), or 0 to cancel:"
    read -r selection

    if [ "$selection" = "0" ]; then
        echo "Setup cancelled."
        exit 0
    fi

    if [ "$selection" -ge 1 ] && [ "$selection" -le $total_devices ]; then
        selected_path="${device_paths[$selection]}"
        selected_name="${device_names[$selection]}"

        echo "Selected: $selected_path ($selected_name)"

        # Set permissions
        echo "Setting permissions for $selected_path..."
        sudo chmod 666 "$selected_path"

        if [ $? -eq 0 ]; then
            echo " Permissions set successfully!"

            # Add to known names for future use
            echo "Adding '$selected_name' to known joystick names..."
            echo "$selected_name" >> "$KNOWN_NAMES_FILE"
            echo " Device name saved for future automatic detection."

            # Test with ROS2 if available
            if command -v ros2 &> /dev/null; then
                echo ""
                echo "Testing ROS2 joystick detection..."
                ros2 run joy joy_enumerate_devices 2>/dev/null || echo "Note: ROS2 joy package may not be available in this environment"
            fi

            echo ""
            echo "Joystick setup completed successfully!"
            echo "Device: $selected_path ($selected_name)"
        else
            echo "Error: Failed to set permissions. Make sure you have sudo access."
            exit 1
        fi
    else
        echo "Invalid selection. Setup cancelled."
        exit 1
    fi
fi
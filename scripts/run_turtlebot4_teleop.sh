#!/usr/bin/env bash

# Start the keyboard controller with the Raspberry Pi's system ROS 2 Python.
# This script deliberately does not use uv: rclpy and geometry_msgs are
# supplied by the ROS installation under /opt/ros, not by PyPI.

set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(dirname -- "$SCRIPT_DIR")"

if [[ -f /etc/turtlebot4/setup.bash ]]; then
    # The official TurtleBot image records its ROS and DDS configuration here.
    source /etc/turtlebot4/setup.bash
elif [[ -n "${ROS_DISTRO:-}" && -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]]; then
    source "/opt/ros/${ROS_DISTRO}/setup.bash"
else
    for distribution in jazzy humble galactic; do
        setup_file="/opt/ros/${distribution}/setup.bash"
        if [[ -f "$setup_file" ]]; then
            source "$setup_file"
            break
        fi
    done
fi

if [[ -z "${ROS_DISTRO:-}" ]]; then
    echo "ROS 2 was not found on this computer." >&2
    echo "Run this script on the TurtleBot 4 Raspberry Pi." >&2
    exit 1
fi

if ! python3 -c 'import rclpy; import geometry_msgs' 2>/dev/null; then
    echo "ROS 2 is set to '${ROS_DISTRO}', but its Python modules are missing." >&2
    echo "Use the Raspberry Pi system python3; do not use uv or a virtualenv." >&2
    echo "Python: $(command -v python3)" >&2
    exit 1
fi

echo "Using ROS 2 ${ROS_DISTRO} with $(command -v python3)"
cd "$REPOSITORY_ROOT"
exec python3 -m turtlebot4_control.keyboard_teleop "$@"


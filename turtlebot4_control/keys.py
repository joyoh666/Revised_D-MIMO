"""Terminal-key mappings kept independent from ROS for easy testing."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VelocityCommand:
    """Planar velocity requested by one keyboard key."""

    linear_x: float
    angular_z: float


def command_for_key(
    key: str,
    linear_speed: float,
    angular_speed: float,
) -> VelocityCommand | None:
    """Translate an arrow key or space into a planar velocity command."""
    commands = {
        "\x1b[A": VelocityCommand(linear_speed, 0.0),
        "\x1b[B": VelocityCommand(-linear_speed, 0.0),
        "\x1b[C": VelocityCommand(0.0, -angular_speed),
        "\x1b[D": VelocityCommand(0.0, angular_speed),
        " ": VelocityCommand(0.0, 0.0),
    }
    return commands.get(key)


#!/usr/bin/env python3
"""Drive a physical TurtleBot 4 Lite with terminal arrow keys.

Run this file on the robot's Raspberry Pi after sourcing its ROS 2 setup.
The publisher automatically uses TwistStamped on Jazzy and Twist on older
ROS 2 distributions. Use --stamped or --unstamped to override that choice.
"""

from __future__ import annotations

import argparse
import os
import select
import sys
import termios
import time
import tty
from contextlib import contextmanager
from typing import Iterator, Sequence

try:
    import rclpy
    from geometry_msgs.msg import Twist, TwistStamped
    from rclpy.node import Node
except ModuleNotFoundError as exc:
    if exc.name in {"rclpy", "geometry_msgs"}:
        raise SystemExit(
            "ROS 2 Python modules were not found. Run this controller on the "
            "TurtleBot Raspberry Pi with:\n"
            "  ./scripts/run_turtlebot4_teleop.sh\n"
            "Do not use 'uv run' for this ROS node."
        ) from None
    raise

from turtlebot4_control.keys import VelocityCommand, command_for_key


HELP = """
TurtleBot 4 Lite keyboard control

  ↑ / ↓       move forward / backward
  ← / →       rotate left / right
  Space       stop immediately
  + / -       increase / decrease both speeds by 10%
  q           stop and quit

Hold an arrow key to keep moving. Releasing it triggers the safety timeout.
Keep the robot in sight and keep a hand near Space or the Create 3 stop button.
"""

MAX_LINEAR_SPEED = 0.31
MAX_ANGULAR_SPEED = 1.9


def _parse_args(
    argv: Sequence[str] | None = None,
) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Drive a TurtleBot 4 Lite with terminal arrow keys."
    )
    parser.add_argument("--topic", default="/cmd_vel")
    parser.add_argument("--linear-speed", type=float, default=0.15)
    parser.add_argument("--angular-speed", type=float, default=0.8)
    parser.add_argument(
        "--timeout",
        type=float,
        default=0.6,
        help="stop this many seconds after the last arrow-key event",
    )
    message_group = parser.add_mutually_exclusive_group()
    message_group.add_argument("--stamped", action="store_true")
    message_group.add_argument("--unstamped", action="store_true")
    args, ros_args = parser.parse_known_args(argv)

    if args.linear_speed <= 0.0:
        parser.error("--linear-speed must be positive")
    if args.linear_speed > MAX_LINEAR_SPEED:
        parser.error(f"--linear-speed must not exceed {MAX_LINEAR_SPEED} m/s")
    if args.angular_speed <= 0.0:
        parser.error("--angular-speed must be positive")
    if args.angular_speed > MAX_ANGULAR_SPEED:
        parser.error(
            f"--angular-speed must not exceed {MAX_ANGULAR_SPEED} rad/s"
        )
    if args.timeout <= 0.0:
        parser.error("--timeout must be positive")
    return args, ros_args


def _use_stamped_message(args: argparse.Namespace) -> bool:
    if args.stamped:
        return True
    if args.unstamped:
        return False
    return os.environ.get("ROS_DISTRO", "").lower() in {
        "jazzy",
        "kilted",
        "rolling",
    }


@contextmanager
def _raw_terminal() -> Iterator[int]:
    if not sys.stdin.isatty():
        raise RuntimeError("keyboard control requires an interactive terminal")
    descriptor = sys.stdin.fileno()
    previous = termios.tcgetattr(descriptor)
    try:
        tty.setcbreak(descriptor)
        yield descriptor
    finally:
        termios.tcsetattr(descriptor, termios.TCSADRAIN, previous)


def _read_key(descriptor: int, timeout: float) -> str | None:
    readable, _, _ = select.select([descriptor], [], [], timeout)
    if not readable:
        return None

    data = os.read(descriptor, 1)
    if data == b"\x1b":
        # Arrow keys arrive as a three-byte ANSI escape sequence.
        for _ in range(2):
            readable, _, _ = select.select([descriptor], [], [], 0.02)
            if not readable:
                break
            data += os.read(descriptor, 1)
    return data.decode(errors="ignore")


class TurtleBot4Teleop(Node):
    """Publish velocity commands using the ROS version's preferred type."""

    def __init__(self, topic: str, stamped: bool) -> None:
        super().__init__("turtlebot4_arrow_teleop")
        self._stamped = stamped
        message_type = TwistStamped if stamped else Twist
        self._publisher = self.create_publisher(message_type, topic, 10)

    def publish(self, command: VelocityCommand) -> None:
        twist = Twist()
        twist.linear.x = command.linear_x
        twist.angular.z = command.angular_z
        if self._stamped:
            message = TwistStamped()
            message.header.stamp = self.get_clock().now().to_msg()
            message.header.frame_id = "base_link"
            message.twist = twist
        else:
            message = twist
        self._publisher.publish(message)

    def stop(self) -> None:
        self.publish(VelocityCommand(0.0, 0.0))


def main(argv: Sequence[str] | None = None) -> None:
    args, ros_args = _parse_args(argv)
    stamped = _use_stamped_message(args)
    rclpy.init(args=ros_args)
    node = TurtleBot4Teleop(args.topic, stamped)
    linear_speed = args.linear_speed
    angular_speed = args.angular_speed
    current = VelocityCommand(0.0, 0.0)
    last_motion_key = 0.0
    moving = False

    node.get_logger().info(
        f"publishing {'TwistStamped' if stamped else 'Twist'} on {args.topic}"
    )
    print(HELP)

    try:
        with _raw_terminal() as descriptor:
            while rclpy.ok():
                key = _read_key(descriptor, timeout=0.05)
                now = time.monotonic()

                if key in {"q", "Q", "\x03"}:
                    break
                if key in {"+", "="}:
                    linear_speed = min(MAX_LINEAR_SPEED, linear_speed * 1.1)
                    angular_speed = min(
                        MAX_ANGULAR_SPEED, angular_speed * 1.1
                    )
                    print(
                        f"\rlinear={linear_speed:.2f} m/s, "
                        f"angular={angular_speed:.2f} rad/s     ",
                        end="",
                        flush=True,
                    )
                elif key in {"-", "_"}:
                    linear_speed = max(0.02, linear_speed / 1.1)
                    angular_speed = max(0.1, angular_speed / 1.1)
                    print(
                        f"\rlinear={linear_speed:.2f} m/s, "
                        f"angular={angular_speed:.2f} rad/s     ",
                        end="",
                        flush=True,
                    )

                command = (
                    command_for_key(key, linear_speed, angular_speed)
                    if key is not None
                    else None
                )
                if command is not None:
                    current = command
                    moving = command != VelocityCommand(0.0, 0.0)
                    last_motion_key = now
                elif moving and now - last_motion_key >= args.timeout:
                    current = VelocityCommand(0.0, 0.0)
                    moving = False

                node.publish(current)
                rclpy.spin_once(node, timeout_sec=0.0)
    except (KeyboardInterrupt, RuntimeError) as exc:
        if isinstance(exc, RuntimeError):
            node.get_logger().error(str(exc))
    finally:
        # Send several zero commands so a single dropped packet cannot leave
        # the base moving when this process exits.
        for _ in range(3):
            node.stop()
            time.sleep(0.05)
        node.destroy_node()
        rclpy.shutdown()
        print("\nStopped.")


if __name__ == "__main__":
    main()

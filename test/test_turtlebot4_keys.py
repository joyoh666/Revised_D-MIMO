from turtlebot4_control.keys import VelocityCommand, command_for_key


def test_arrow_keys_map_to_turtlebot_planar_velocities():
    assert command_for_key("\x1b[A", 0.15, 0.8) == VelocityCommand(0.15, 0.0)
    assert command_for_key("\x1b[B", 0.15, 0.8) == VelocityCommand(-0.15, 0.0)
    assert command_for_key("\x1b[C", 0.15, 0.8) == VelocityCommand(0.0, -0.8)
    assert command_for_key("\x1b[D", 0.15, 0.8) == VelocityCommand(0.0, 0.8)


def test_space_stops_and_unrecognised_keys_are_ignored():
    assert command_for_key(" ", 0.15, 0.8) == VelocityCommand(0.0, 0.0)
    assert command_for_key("x", 0.15, 0.8) is None


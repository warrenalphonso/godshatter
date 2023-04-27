from dataclasses import dataclass


@dataclass
class Action:
    direction: str
    magnitude: int


with open("data/2.txt") as f:
    lines = f.read().rstrip().split("\n")
    actions: list[Action] = []
    for line in lines:
        direction, magnitude = line.split(" ")
        assert direction in ["forward", "up", "down"]
        actions.append(Action(direction=direction, magnitude=int(magnitude)))


def one():
    """
    Interpretation of the actions:
    - "forward X": increase horizontal position by X
    - "down X": increase depth by X
    - "up X": decrease depth by X
    Multiply your final horizontal position by your final depth.
    """
    horizontal, depth = 0, 0
    for action in actions:
        if action.direction == "forward":
            horizontal += action.magnitude
        elif action.direction == "up":
            depth -= action.magnitude
        else:
            depth += action.magnitude

    return horizontal * depth


def two():
    """
    Interpretation of the actions:
    - "forward X": increase horizontal position by X, increase depth by aim * X
    - "down X": increase aim by X
    - "up X": decrease aim by X
    Multiply your final horizontal position by your final depth.
    """
    horizontal, depth, aim = 0, 0, 0
    for action in actions:
        if action.direction == "forward":
            horizontal += action.magnitude
            depth += aim * action.magnitude
        elif action.direction == "up":
            aim -= action.magnitude
        else:
            aim += action.magnitude

    return horizontal * depth


if __name__ == "__main__":
    print(f"Solution for Part 1: {one()}")
    print(f"Solution for Part 2: {two()}")

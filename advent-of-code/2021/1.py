from more_itertools import windowed

with open("data/1.txt") as f:
    measurements = [int(n) for n in f.read().rstrip().split("\n")]


def one():
    """How many measurements are larger than the previous measurement?"""
    return sum(i < j for i, j in windowed(measurements, 2))


def two():
    """
    How many times does the sum of measurements in a sliding three-window
    increase?
    """
    three_sums = [i + j + k for i, j, k in windowed(measurements, 3)]
    return sum(i < j for i, j in windowed(three_sums, 2))


if __name__ == "__main__":
    print(f"Solution for Part 1: {one()}")
    print(f"Solution for Part 2: {two()}")

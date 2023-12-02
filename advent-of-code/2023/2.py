import pathlib
import re
from collections import Counter, namedtuple
from math import prod

Draw = namedtuple("Draw", ["red", "green", "blue"])

lines = (pathlib.Path(__file__).parent / "data/2.txt").read_text().strip().split("\n")
data = []
for line in lines:
    # Remove "Game N: " prefix
    line = re.sub(r"Game \d+: ", "", line)
    draws = line.split(";")
    results = []
    for draw in draws:
        colors = draw.split(",")
        result = {}
        for color in colors:
            color = color.strip()
            num, color_ = color.split(" ")
            result[color_] = int(num)
        results.append(result)

    data.append(
        tuple(
            Draw(red=result.get("red", 0), green=result.get("green", 0), blue=result.get("blue", 0))
            for result in results
        )
    )


def one():
    """Which games are possible with 12 red cubes, 13 green cubes, and 14 blue cubes?"""
    constraint = Counter(red=12, green=13, blue=14)
    valid_games = []
    for i, game in enumerate(data):
        for draw in game:
            for color in ["red", "green", "blue"]:
                if getattr(draw, color) > constraint[color]:
                    break
            else:
                continue
            break
        else:
            valid_games.append(i + 1)

    return sum(valid_games)


def two():
    """What is the minimum number of cubes in the set that makes each game possible?"""
    mins = []
    for game in data:
        bag = Counter()
        for draw in game:
            for color in ["red", "green", "blue"]:
                bag[color] = max(bag[color], getattr(draw, color))
        mins.append(bag)

    return sum(bag["red"] * bag["green"] * bag["blue"] for bag in mins)

from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class Cave:
    name: str
    connections: list["Cave"]

    @property
    def size(self):
        return "small" if self.name.islower() else "big"


with open("data/12.txt") as f:
    connections = f.read().rstrip().split("\n")

    caves: list[Cave] = []
    for connection in connections:
        from_name, to_name = connection.split("-")

        try:
            from_cave = next(cave for cave in caves if cave.name == from_name)
        except StopIteration:
            from_cave = Cave(name=from_name, connections=[])
            caves.append(from_cave)

        try:
            to_cave = next(cave for cave in caves if cave.name == to_name)
        except StopIteration:
            to_cave = Cave(name=to_name, connections=[])
            caves.append(to_cave)

        from_cave.connections.append(to_cave)
        to_cave.connections.append(from_cave)


def one():
    """
    How many paths are there from start to end that don't visit small caves
    multiple times?

    I don't think there can be repeated paths, so just doing DFS should be fine
    without checking for redundancy.
    """

    # We have to track some state per path, so let's do it using a function argument
    def paths(start: Cave, end: Cave, smalls_visited: list[str]) -> list[list[str]]:
        """Return list of paths from start to end from each of start's connections."""
        # Base case
        if start.name == end.name:
            # One path: do nothing
            return [[]]

        if start.size == "small":
            smalls_visited.append(start.name)

        connection_paths: dict[str, list[list[str]]] = {}
        for connection in start.connections:
            if connection.name not in smalls_visited:
                copy = smalls_visited.copy()
                connection_paths[connection.name] = paths(connection, end, copy)
        all_paths: list[list[str]] = []
        for connection, paths_ in connection_paths.items():
            for path in paths_:
                all_paths.append([connection, *path])

        return all_paths

    start = next(cave for cave in caves if cave.name == "start")
    end = next(cave for cave in caves if cave.name == "end")

    return len(paths(start, end, []))


def two():
    """Same as one but now we can visit one small cave twice, except start and end."""

    def can_visit_cave(cave: Cave, smalls_visited: list[str]):
        if cave.size == "big" or cave.name not in smalls_visited:
            return True
        if cave.name in ("start", "end"):
            return False
        # Check if we've already visited some cave twice
        return Counter(smalls_visited).most_common(1)[0][1] < 2

    # We have to track some state per path, so let's do it using a function argument
    def paths(start: Cave, end: Cave, smalls_visited: list[str]) -> list[list[str]]:
        """Return list of paths from start to end from each of start's connections."""
        # Base case
        if start.name == end.name:
            # One path: do nothing
            return [[]]

        if start.size == "small":
            smalls_visited.append(start.name)

        connection_paths: dict[str, list[list[str]]] = {}
        for connection in start.connections:
            if can_visit_cave(connection, smalls_visited):
                copy = smalls_visited.copy()
                connection_paths[connection.name] = paths(connection, end, copy)
        all_paths: list[list[str]] = []
        for connection, paths_ in connection_paths.items():
            for path in paths_:
                all_paths.append([connection, *path])

        return all_paths

    start = next(cave for cave in caves if cave.name == "start")
    end = next(cave for cave in caves if cave.name == "end")

    return len(paths(start, end, []))


if __name__ == "__main__":
    print(f"Solution for Part 1: {one()}")
    print(f"Solution for Part 2: {two()}")

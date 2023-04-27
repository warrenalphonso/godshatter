from dataclasses import dataclass


@dataclass
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
    """

    def paths(start: Cave, end: Cave, smalls_visited: list[str] = []) -> list[int]:
        """Return list of number of paths from start to end from each of start's connections."""
        if start.size == "small":
            smalls_visited.append(start.name)
        return [
            paths(connection, end, smalls_visited)
            for connection in start.connections
            if connection.name not in smalls_visited
        ]

    start = next(cave for cave in caves if cave.name == "start")
    end = next(cave for cave in caves if cave.name == "end")

    return paths(start, end, [])


def two():
    pass


if __name__ == "__main__":
    print(f"Solution for Part 1: {one()}")
    print(f"Solution for Part 2: {two()}")

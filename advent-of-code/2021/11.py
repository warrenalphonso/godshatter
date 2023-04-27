# We need a 2D grid and the ability to get neighbors, including diagonal
# neighbors. We don't wrap around (index -1) to find neighbors. Norvig's 1D
# dictionary with keys tuples of (x,y) works well for this.
class Grid(dict):
    def neighbors(self, coords) -> list[int]:
        x, y = coords
        return [
            (x + i, y + j)
            for i in (-1, 0, 1)
            for j in (-1, 0, 1)
            if (i != 0 or j != 0) and (x + i, y + j) in self
        ]


with open("data/11.txt") as f:
    lines = f.read().rstrip().split("\n")
    grid = Grid()
    for x in range(len(lines)):
        for y in range(len(lines[0])):
            grid[(x, y)] = int(lines[x][y])


def one():
    """
    At each time step, increase all energy levels by 1. Any levels above 9 should
    *flash*, which increases the energy level of all its neighbors by 1. Those
    neighbors might flash also. Each octopus can only flash once per step.
    All octopuses with energy level above 9 must reset to 0 at the next step.
    How many flashes occur after 100 steps?
    """
    flashes = 0
    for _ in range(100):
        # Increase all energy levels by 1
        for coords in grid:
            grid[coords] += 1

        # While octopuses that haven't flashed have energy above 9, flash
        flashed_this_step = set()
        while True:
            updated_this_step = False
            for coords, energy_level in grid.items():
                if energy_level > 9 and coords not in flashed_this_step:
                    flashed_this_step.add(coords)
                    flashes += 1
                    updated_this_step = True
                    for n_coords in grid.neighbors(coords):
                        grid[n_coords] += 1
            if not updated_this_step:
                break

        # Set all octopuses with energy above 9 to energy
        for coords, energy_level in grid.items():
            if energy_level > 9:
                grid[coords] = 0

    return flashes


def two():
    """
    What's the first step during which all octopuses flash simultaneously?
    """
    i = 0
    while True:
        i += 1

        # Increase all energy levels by 1
        for coords in grid:
            grid[coords] += 1

        # While octopuses that haven't flashed have energy above 9, flash
        flashed_this_step = set()

        while True:
            updated_this_step = False
            for coords, energy_level in grid.items():
                if energy_level > 9 and coords not in flashed_this_step:
                    flashed_this_step.add(coords)
                    updated_this_step = True
                    for n_coords in grid.neighbors(coords):
                        grid[n_coords] += 1
            if not updated_this_step:
                break

        # Did all octopuses flash simultaneously?
        if len(flashed_this_step) == len(grid):
            return i

        # Set all octopuses with energy above 9 to energy
        for coords, energy_level in grid.items():
            if energy_level > 9:
                grid[coords] = 0


if __name__ == "__main__":
    # print(f"Solution for Part 1: {one()}")
    print(f"Solution for Part 2: {two()}")

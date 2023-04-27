with open("data/9.txt") as f:
    lines = f.read().rstrip().split("\n")
    grid: list[list[int]] = []
    for line in lines:
        grid.append([int(n) for n in line])


# TODO: From Norvig, I realized a cleaner way to get this behavior is to define
# the grid as a 1D dictionary with keys (x, y). Then to see if a neighbor is
# valid, like (-1, 2) I can just check if this literal key is in the dictionary.
# Nice way to prevent wrap-arounds.
def neighbors(grid, x, y):
    if x - 1 >= 0:
        yield (x - 1, y)
    if x + 1 < len(grid):
        yield (x + 1, y)
    if y - 1 >= 0:
        yield (x, y - 1)
    if y + 1 < len(grid[0]):
        yield (x, y + 1)


def one():
    """
    Find the points in the grid that are the lower than their neighbors.
    Their "risk level" is 1 + their height. Sum the risk levels of these points.
    """
    risk_level = 0
    for x in range(len(grid)):
        for y in range(len(grid[0])):
            for n_x, n_y in neighbors(grid, x, y):
                if grid[x][y] >= grid[n_x][n_y]:
                    break
            else:
                risk_level += grid[x][y] + 1
    return risk_level


def two():
    """
    A basin is all locations that eventually flow downward to a single low point.
    Locations of height 9 don't count as being in any basin.
    What's the product of the sizes of the three largest basins?

    Strategy: maintain a list of areas of the basins. We go through the grid
    until we find a number that isn't 9. Then, we do a DFS to find the size of
    this basin, mutating the grid to change it to 9 after we count it.
    """

    def basin_size(grid, x, y):
        """
        If grid[x][y], this is a basin of size 0.
        Otherwise recursively calculate the size, setting grid[i][j] = 9 once
        it's counted.
        """
        if grid[x][y] == 9:
            return 0
        grid[x][y] = 9
        return 1 + sum(basin_size(grid, n_x, n_y) for n_x, n_y in neighbors(grid, x, y))

    basin_sizes = []
    for x in range(len(grid)):
        for y in range(len(grid[0])):
            basin_sizes.append(basin_size(grid, x, y))
    basin_sizes = sorted(basin_sizes)
    return basin_sizes[-3] * basin_sizes[-2] * basin_sizes[-1]


if __name__ == "__main__":
    print(f"Solution for Part 1: {one()}")
    print(f"Solution for Part 2: {two()}")

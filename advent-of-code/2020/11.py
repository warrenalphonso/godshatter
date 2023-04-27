from helpers import data

# "L": empty seat
# "#": occupied seat
# ".": floor
rows: list[str] = data(11)

# Part 1


# These rules are applied simultaneously to all seats:
# 1. If a seat is empty and no occupied seats adjacent, it becomes occupied
# 2. If a seat is occupied with 4 or more occupied seats adjacent, it becomes empty
# Keep applying the rules until the seats stabilize.

# Approach: since we want to simultaneously apply the rules, let's creates a new
# array of strings for the new state. Then we can compare it to the old state.


# We only care about the total states of adjacent seats, so let's ignore their
# actual positions
def get_adjacent_seats(rows: list[str], row: int, col: int) -> str:
    """
    Return a string of all adjacent seats to seat at (row, col).

    >>> rows = ["LLL", ".#L", "..."]
    >>> get_adjacent_seats(rows, 0, 0)
    'L.#'
    >>> get_adjacent_seats(rows, 1, 1)
    'LLL.L...'
    >>> get_adjacent_seats(rows, 2, 1)
    '.#L..'
    """
    adjacent = ""
    for row_offset in (-1, 0, 1):
        for col_offset in (-1, 0, 1):
            if (row_offset, col_offset) == (0, 0):
                # Current seat isn't adjacent
                continue

            adjacent_row = row + row_offset
            adjacent_col = col + col_offset
            # Don't wrap around
            if (
                adjacent_row < 0
                or adjacent_row >= len(rows)
                or adjacent_col < 0
                or adjacent_col >= len(rows[0])
            ):
                continue

            adjacent += rows[adjacent_row][adjacent_col]

    return adjacent


def stable_adjacent(rows):
    while True:
        next_state = []
        for i in range(len(rows)):
            next_state.append("")

            for j in range(len(rows[0])):
                adjacent_seats = get_adjacent_seats(rows, i, j)
                occupied = sum(c == "#" for c in adjacent_seats)
                if rows[i][j] == "L" and occupied == 0:
                    next_state[i] += "#"
                    continue
                elif rows[i][j] == "#" and occupied >= 4:
                    next_state[i] += "L"
                    continue
                next_state[i] += rows[i][j]

        if next_state == rows:
            return sum(c == "#" for c in "".join(rows))
        rows = next_state


# Part 2


def get_adjcent_line_of_sight(rows, row, col):
    adjacent = ""
    for row_offset in (-1, 0, 1):
        for col_offset in (-1, 0, 1):
            if (row_offset, col_offset) == (0, 0):
                # Current seat isn't adjacent
                continue

            adjacent_row, adjacent_col = row + row_offset, col + col_offset

            # Keep going until we find a seat or hit the edge
            while True:
                # Don't wrap around
                if (
                    adjacent_row < 0
                    or adjacent_row >= len(rows)
                    or adjacent_col < 0
                    or adjacent_col >= len(rows[0])
                ):
                    break

                # If it's a floor, keep looking in same direction
                seat = rows[adjacent_row][adjacent_col]
                if seat == ".":
                    adjacent_row += row_offset
                    adjacent_col += col_offset
                    continue

                adjacent += seat
                break

    return adjacent


def stable_line_of_sight(rows):
    while True:
        next_state = []
        for i in range(len(rows)):
            next_state.append("")

            for j in range(len(rows[0])):
                adjacent_seats = get_adjcent_line_of_sight(rows, i, j)
                occupied = sum(c == "#" for c in adjacent_seats)
                if rows[i][j] == "L" and occupied == 0:
                    next_state[i] += "#"
                    continue
                elif rows[i][j] == "#" and occupied >= 5:
                    next_state[i] += "L"
                    continue
                next_state[i] += rows[i][j]

        if next_state == rows:
            return sum(c == "#" for c in "".join(rows))
        rows = next_state


if __name__ == "__main__":
    import doctest

    doctest.testmod()

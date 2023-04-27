from collections import defaultdict, namedtuple


class Point(complex):
    # Shout-out Norvig: https://norvig.com/python-iaq.html
    @property
    def x(self):
        return self.real

    @property
    def y(self):
        return self.imag


def pos_neg_range(n):
    """
    ``range``, but from 0 to n, with stride 1, assuming n is an integer.

    >>> list(pos_neg_range(4))
    [0, 1, 2, 3, 4]
    >>> list(pos_neg_range(-4))
    [0, -1, -2, -3, -4]
    """
    if n > 0:
        return range(int(n) + 1)
    return range(0, int(n) - 1, -1)


with open("data/5.txt") as f:
    lines = []
    for line in f.read().rstrip().split("\n"):
        start, end = line.split(" -> ")
        start_x, start_y = [int(n) for n in start.split(",")]
        end_x, end_y = [int(n) for n in end.split(",")]
        lines.append((Point(start_x, start_y), Point(end_x, end_y)))


def one():
    not_diagonal = [line for line in lines if line[0].x == line[1].x or line[0].y == line[1].y]
    c = defaultdict(int)
    for start, end in not_diagonal:
        diff = Point(end - start)
        if diff.x:
            for i in pos_neg_range(diff.x):
                point = Point(start.x + i, start.y)
                c[point] += 1
        else:
            for j in pos_neg_range(diff.y):
                point = Point(start.x, start.y + j)
                c[point] += 1

    return sum(1 for v in c.values() if v >= 2)


def two():
    c = defaultdict(int)
    for start, end in lines:
        diff = Point(end - start)
        if diff.x and diff.y:
            for i in pos_neg_range(diff.x):
                if (diff.x * diff.y) > 0:
                    # Same sign
                    point = Point(start.x + i, start.y + i)
                else:
                    point = Point(start.x + i, start.y - i)
                c[point] += 1
        elif diff.x:
            for i in pos_neg_range(diff.x):
                point = Point(start.x + i, start.y)
                c[point] += 1
        else:
            for j in pos_neg_range(diff.y):
                point = Point(start.x, start.y + j)
                c[point] += 1

    return sum(1 for v in c.values() if v >= 2)


if __name__ == "__main__":
    from doctest import testmod

    testmod()
    print(two())

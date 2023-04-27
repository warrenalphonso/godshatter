with open("data/7.txt") as f:
    crab_positions = [int(n) for n in f.read().rstrip().split(",")]


def one():
    min_pos, max_pos = min(crab_positions), max(crab_positions)

    min_cost = None
    for pos in range(min_pos, max_pos + 1):
        cost = 0
        for crab_pos in crab_positions:
            cost += abs(crab_pos - pos)
        if min_cost is None or cost < min_cost:
            min_cost = cost
    return min_cost


def two():
    min_pos, max_pos = min(crab_positions), max(crab_positions)
    move_cost = {0: 0}
    for move in range(1, max_pos - min_pos + 1):
        move_cost[move] = move**2 - move_cost[move - 1]

    min_cost = None
    for pos in range(min_pos, max_pos + 1):
        cost = 0
        for crab_pos in crab_positions:
            move = abs(crab_pos - pos)
            cost += move_cost[move]
        if min_cost is None or cost < min_cost:
            min_cost = cost
    return min_cost


if __name__ == "__main__":
    print(two())

# 1 = 1
# 1 + 2 = 3
# 1 + 2 + 3 = 6
# 1 + 2 + 3 + 4 = 10
# 1 + 2 + 3 + 4 + 5 = 15

# 1**2 - 0
# 2**2 - 1
# 3**2 - 3
# 4**2 - 6
# 5**2 - 10

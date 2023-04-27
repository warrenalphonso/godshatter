from helpers import data

ready_time, bus_ids = data(13)
ready_time = int(ready_time)
bus_ids = [int(id) if id != "x" else id for id in bus_ids.split(",")]

# Part 1
# The bus IDs are how often a bus departs from the sea port. Which is the earliest
# bus we can take? Multiply the ID by the number of minutes we have to wait.


def smallest_multiple_greater_than(n, k):
    """Return the smallest multiple of k greater than n."""
    if n % k == 0:
        return n
    return k * (n // k + 1)


def solve_one():
    # Assume the buses with ID "x" are out-of-service
    valid_bus_ids = [id for id in bus_ids if id != "x"]
    earliest = [smallest_multiple_greater_than(ready_time, k) for k in valid_bus_ids]
    return (min(earliest) - ready_time) * valid_bus_ids[earliest.index(min(earliest))]


# Part 2
# Find the earliest time such that the first ID in our list departs at that time,
# the second ID departs at the minute after, the third departs two minutes after,
# and so on. Bus IDs with "x" have no constraints.
#
# Approach:
# Find some N such that N mod id_0 == 0, N mod id_1 == 1, ...
# We could find the largest bus ID, keep trying multiples of it and checking
# against the mod of other IDs in decreasing order.

from collections import Counter

with open("data/3.txt") as f:
    diagnostic_report = f.read().rstrip().split("\n")


def one():
    """
    The "gamma rate" is a binary number where each bit is the most common bit
    in the corresponding position of all numbers in the diagnostic report.
    The "epsilon rate" is a binary number where each bit is the least common bit
    for the position over all numbers in the diagnostic report.
    The "power consumption" is the "gamma rate" * "epsilon rate".
    What's the power consumption?
    """
    gamma_rate, epsilon_rate = "", ""
    for bits in zip(*diagnostic_report):
        count = Counter(bits)
        most_common, least_common = count.most_common()
        gamma_rate += most_common[0]
        epsilon_rate += least_common[0]
    return int(gamma_rate, base=2) * int(epsilon_rate, base=2)


def two():
    """
    The "oxygen generator rating" is a binary number produced by successively
    keeping only the numbers with the most common bit in each position,
    resolving ties with "1".
    The "CO2 scrubber rating" is a binary number produced by successively keeping
    only the numbers with the least common bit in each position, resolving ties
    with "0".
    The "life support rating" is the "oxygen generator rating" * "CO2 scrubber rating".
    What's the life support rating?
    """
    potential_O2_ratings = diagnostic_report.copy()
    for pos in range(len(diagnostic_report[0])):
        count = Counter(num[pos] for num in potential_O2_ratings)
        most_common, least_common = count.most_common()
        tie = most_common[1] == least_common[1]
        target = most_common[0] if not tie else "1"
        potential_O2_ratings = [num for num in potential_O2_ratings if num[pos] == target]
        if len(potential_O2_ratings) == 1:
            break

    potential_CO2_ratings = diagnostic_report.copy()
    for pos in range(len(diagnostic_report[0])):
        count = Counter(num[pos] for num in potential_CO2_ratings)
        most_common, least_common = count.most_common()
        tie = most_common[1] == least_common[1]
        target = least_common[0] if not tie else "0"
        potential_CO2_ratings = [num for num in potential_CO2_ratings if num[pos] == target]
        if len(potential_CO2_ratings) == 1:
            break

    return int(potential_O2_ratings[0], base=2) * int(potential_CO2_ratings[0], base=2)


if __name__ == "__main__":
    print(f"Solution for Part 1: {one()}")
    print(f"Solution for Part 2: {two()}")

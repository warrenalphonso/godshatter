from dataclasses import dataclass

# Order of segments doesn't matter
Digit = set

ZERO = Digit("abcefg")
ONE = Digit("cf")
TWO = Digit("acdeg")
THREE = Digit("acdfg")
FOUR = Digit("bcdf")
FIVE = Digit("abdfg")
SIX = Digit("abdefg")
SEVEN = Digit("acf")
EIGHT = Digit("abcdefg")
NINE = Digit("abcdfg")


def decimal(digit: Digit):
    if digit == ZERO:
        return "0"
    elif digit == ONE:
        return "1"
    elif digit == TWO:
        return "2"
    elif digit == THREE:
        return "3"
    elif digit == FOUR:
        return "4"
    elif digit == FIVE:
        return "5"
    elif digit == SIX:
        return "6"
    elif digit == SEVEN:
        return "7"
    elif digit == EIGHT:
        return "8"
    elif digit == NINE:
        return "9"
    else:
        return "0"
        raise ValueError(f"{digit} is unrecognized!")


@dataclass
class Display:
    digits: list[Digit]
    output: list[Digit]


with open("data/8.txt") as f:
    lines = f.read().rstrip().split("\n")

    displays: list[Display] = []
    for line in lines:
        digits, output = line.split(" | ")
        digits = [Digit(segments) for segments in digits.split(" ")]
        output = [Digit(segments) for segments in output.split(" ")]
        displays.append(Display(digits, output))


def one():
    """
    The digits 1, 4, 7, and 8 each use a unique number of segments.
    How many digits in the output values are one of these digits?
    """
    return sum(
        len(digit) in (len(ONE), len(FOUR), len(SEVEN), len(EIGHT))
        for display in displays
        for digit in display.output
    )


def two():
    """
    What's the sum of the decoded output values?

    Strategy:
    - We know which digits are meant to be 1, 4, 7, and 8.
    - Find the segment in 7 but not in 1. That's "a".
    - Find the segments in 4 but not in 1. That's "bd".
    - The only five-segment digit with "abd" is 5. 5 has two other segments:
      "fg". Whichever one of these isn't in 1 must be "g". The one which is must
      be "f". The other segment in 1 must be "c".
    - Now we know "a", "c", "f", and "g".
    - 3 - "acfg" gives us "d".
    - 2 - "acdg" gives us "e".
    - 8 - "acdefg" gives us "b".
    """
    total = 0
    for display in displays:
        mapping = {}

        # Figure out which digits are 1, 4, 7, 8
        one = next(digit for digit in display.digits if len(digit) == len(ONE))
        four = next(digit for digit in display.digits if len(digit) == len(FOUR))
        seven = next(digit for digit in display.digits if len(digit) == len(SEVEN))
        eight = next(digit for digit in display.digits if len(digit) == len(EIGHT))

        # 7 has segments "acf" and 1 has segments "cf"
        assert len(seven - one) == 1
        mapping["a"] = list(seven - one)[0]

        # 4 has segments "bcdf" and 1 has segments "cf"
        assert len(four - one) == 2
        bd = list(four - one)
        # 5 is the only five-segment digit with "abd". It also has "fg".
        abd = set([mapping["a"], *bd])
        five = next(digit for digit in display.digits if len(digit) == 5 and abd.issubset(digit))
        assert len(five - abd) == 2
        fg = list(five - abd)

        # Whichever segment in fg is in 1 must be "f". The other must be "g".
        # The segment in 1 that isn't in fg must be "c".
        for c in fg:
            if c in one:
                mapping["f"] = c
            else:
                mapping["g"] = c
        for c in one:
            if c not in fg:
                mapping["c"] = c

        # 3 is the only five-segment digit with "acfg". Its fifth segment is "d".
        acfg = {mapping[c] for c in "acfg"}
        three = next(digit for digit in display.digits if len(digit) == 5 and acfg.issubset(digit))
        assert len(three - acfg) == 1
        mapping["d"] = list(three - acfg)[0]

        # 2 is the only five-segment digit with "acdg" aside from 3 which has "f".
        # Its fifth segment is "e".
        acdg = {mapping[c] for c in "acdg"}
        two = next(
            digit
            for digit in display.digits
            if len(digit) == 5 and acdg.issubset(digit) and mapping["f"] not in digit
        )
        assert len(two - acdg) == 1
        mapping["e"] = list(two - acdg)[0]

        # We're only missing "b" and 8 has all the other characters
        mapping["b"] = list(eight - {mapping[c] for c in "acdefg"})[0]

        # Reverse-mapping from encoded-segments to real-segments
        mapping = {v: k for k, v in mapping.items()}
        decoded_output = [Digit(map(lambda c: mapping[c], digit)) for digit in display.output]
        total += int("".join([decimal(digit) for digit in decoded_output]))

    return total


if __name__ == "__main__":
    print(f"Solution for Part 1: {one()}")
    print(f"Solution for Part 2: {two()}")

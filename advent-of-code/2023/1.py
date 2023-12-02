import pathlib
import re
import string

data = (pathlib.Path(__file__).parent / "data/1.txt").read_text().strip().split("\n")


def one():
    """Sum first and last digit in each string in data."""
    calibration_values = []
    for s in data:
        # Strip non-digits
        ss = re.sub(r"\D", "", s)
        assert len(ss) >= 1, s
        calibration_values.append(int(f"{ss[0]}{ss[-1]}"))

    return sum(calibration_values)


def two():
    """Consider spelled-out, lowercase digits valid."""
    spelled_digits = {
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
    }
    pattern = r"\d|" + "|".join(spelled_digits.keys())

    def replace(s):
        if s in string.digits:
            return s
        return spelled_digits[s]

    calibration_values = []
    for s in data:
        # Check for spelled-out digits and map to digits
        # We want to do it greedily left-to-right, so this doesn't work
        # ss = s
        # for spelling, digit in spelled_digits.items():
        #     ss = re.sub(spelling, digit, ss)
        # This doesn't work either because if eg the final characters are "eightwo"
        # then our final digit will be 8.
        # ss = re.sub(pattern, replace, s)
        digits = []
        match = re.search(pattern, s)
        while match is not None:
            s = s[match.start() + 1 :]
            digits.append(replace(match.group(0)))
            match = re.search(pattern, s)

        assert len(digits) >= 1, (s, digits)
        calibration_values.append(int(f"{digits[0]}{digits[-1]}"))

    return sum(calibration_values)

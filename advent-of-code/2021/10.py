"""
There are four types of chunks:
- 0: opened with ( and closed with )
- 1: opened with [ and closed with ]
- 2: opened with { and closed with }
- 3: opened with < and closed with >

When we encounter an opening character, we'll push its chunk type to a stack.
When we encounter a closing character, we'll pop from the stack and ensure the
popped value matches.
"""
with open("data/10.txt") as f:
    lines = f.read().rstrip().split("\n")

CLOSER_TO_OPENER = {")": "(", "]": "[", "}": "{", ">": "<"}
OPENER_TO_COSER = {v: k for k, v in CLOSER_TO_OPENER.items()}


def total_syntax_error_score(first_errors: list[str]) -> int:
    """
    The total syntax score for a set of lines is the sum of error scores for the
    first illegal character on each line.
    """
    total = 0
    for first_error in first_errors:
        if first_error == ")":
            total += 3
        elif first_error == "]":
            total += 57
        elif first_error == "}":
            total += 1197
        elif first_error == ">":
            total += 25137
        else:
            raise ValueError(f"Unrecognized closer: {first_error}")
    return total


def one():
    """
    Consider only the corrupted lines and not the incomplete lines. What's their
    total syntax error score?
    """
    stack = []
    first_errors_corrupted = []
    for line in lines:
        for char in line:
            if char in OPENER_TO_COSER.keys():
                stack.append(char)
            else:
                opener = CLOSER_TO_OPENER[char]
                if stack.pop() != opener:
                    first_errors_corrupted.append(char)
                    break
        stack = []
    return total_syntax_error_score(first_errors_corrupted)


def completion_score(completion_string: list[str]) -> int:
    total = 0
    for closer in completion_string:
        total *= 5
        if closer == ")":
            total += 1
        elif closer == "]":
            total += 2
        elif closer == "}":
            total += 3
        elif closer == ">":
            total += 4
        else:
            raise ValueError(f"Unrecognized closer: {closer}")
    return total


def two():
    """
    Complete the incomplete lines. The completion score i
    """
    stack = []
    incomplete_scores = []
    for line in lines:
        for char in line:
            if char in OPENER_TO_COSER.keys():
                stack.append(char)
            else:
                opener = CLOSER_TO_OPENER[char]
                if stack.pop() != opener:
                    # A corrupted line!
                    break
        else:
            completion_string = [OPENER_TO_COSER[opener] for opener in reversed(stack)]
            incomplete_scores.append(completion_score(completion_string))
        stack = []
    incomplete_scores.sort()
    return incomplete_scores[len(incomplete_scores) // 2]


if __name__ == "__main__":
    print(f"Solution for Part 1: {one()}")
    print(f"Solution for Part 2: {two()}")

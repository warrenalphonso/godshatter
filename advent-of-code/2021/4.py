with open("data/4.txt") as f:
    draw, *boards = f.read().rstrip().split("\n\n")

    draw = [int(n) for n in draw.split(",")]
    empty_boards = [
        [[int(n) for n in row.split()] for row in board.split("\n")] for board in boards
    ]


class Board:
    """A stateful Bingo board."""

    def __init__(self, numbers: list[list[int]]):
        self.numbers = numbers
        self.state = [[0 for _ in range(len(self.numbers[0]))] for _ in range(len(self.numbers))]
        self.finished = False
        self.score = -1
        # Mapping of number to position
        self.positions = {}
        for i in range(len(self.numbers)):
            for j in range(len(self.numbers[0])):
                self.positions[self.numbers[i][j]] = (i, j)

    def add(self, number: int):
        """Check if ``number`` is on Board, and cross it out if it is."""
        if self.finished:
            raise ValueError("This Bingo board has already won.")
        if number in self.positions:
            i, j = self.positions[number]
            self.state[i][j] = 1
            if self._just_completed(i, j):
                self.finished = True
                self.score = self._score(number)

    def _just_completed(self, i: int, j: int) -> bool:
        """Check if filling in ``self.state[i][j]`` completed the game for this Board."""
        row_complete = all(self.state[i][k] for k in range(len(self.state)))
        col_complete = all(self.state[k][j] for k in range(len(self.state[0])))
        return row_complete or col_complete

    def _score(self, last_drawn: int) -> int:
        """The final score is the sum of all unmarked numbers on a board multiplied by the final draw."""
        sum_unmarked = sum(
            self.numbers[i][j]
            for i in range(len(self.numbers))
            for j in range(len(self.numbers[i]))
            if not self.state[i][j]
        )
        return sum_unmarked * last_drawn


def one():
    """What's the final score of the board that wins first?"""
    boards = [Board(numbers=board) for board in empty_boards]
    for n in draw:
        for board in boards:
            board.add(n)
            if board.finished:
                return board.score


def two():
    """What's the final score of the board that wins last?"""
    boards = [Board(numbers=board) for board in empty_boards]
    for n in draw:
        for board in boards:
            board.add(n)

        if len(boards) == 1 and boards[0].finished:
            return boards[0].score

        boards = [board for board in boards if not board.finished]


if __name__ == "__main__":
    print(f"Solution for Part 1: {one()}")
    print(f"Solution for Part 2: {two()}")

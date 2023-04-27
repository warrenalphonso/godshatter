def data(day: int, parser=str, sep="\n") -> list:
    """Split day's input file into sections separated by `sep`, and apply
    parser to each. From Norvig."""
    with open(f"data/{day}.txt") as f:
        sections = f.read().rstrip().split(sep)
        return list(map(parser, sections))

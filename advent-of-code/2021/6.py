from collections import defaultdict

with open("data/6.txt") as f:
    timers = [int(n) for n in f.read().rstrip().split(",")]


def one():
    # 80 days later
    for _ in range(80):
        for i in range(len(timers)):
            timers[i] -= 1
            if timers[i] == -1:
                # Produce new lanternfish with timer 8 for first cycle
                timers[i] = 6
                timers.append(8)
    return len(timers)


def two():
    fish = defaultdict(int)
    for timer in timers:
        fish[timer] += 1

    for _ in range(256):
        new_fish = fish[0]
        for i in range(1, 9):
            fish[i - 1] = fish[i]
        fish[8] = new_fish
        fish[6] += new_fish  # Parents requeue
    return sum(fish.values())


if __name__ == "__main__":
    print(two())

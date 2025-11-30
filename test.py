def compute_sum(n: int) -> int:
    """Return the sum of integers from 1 to n (inclusive)."""
    total = 0
    for i in range(1, n + 1):
        total += i
    return total


if __name__ == '__main__':
    print(compute_sum(10))

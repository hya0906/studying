#factorial(N) = N * factorial(N - 1)
#...
#factorial(1) = 1

def factorial(n):
    if n == 1:
        return 1
    return n * factorial(n - 1)


print(factorial(5))
#페이스북 기출문제 - 어려웠음
input = [0, 3, 5, 6, 1, 2, 4]

def find_max_plus_or_multiply(array): # -> O(N) 시간복잡도
    multiply_sum = 0
    for number in array:
        if number <= 1 or multiply_sum <= 1:
            multiply_sum += number
        else:
            multiply_sum *= number
    return multiply_sum


result = find_max_plus_or_multiply(input)
print(result)
finding_target = 2
finding_numbers = [0, 3, 5, 6, 1, 2, 4]

#이진탐색은 꼭 정렬을 해야한다.
def is_exist_target_number_binary(target, numbers):
    numbers = sorted(numbers)
    current_min = 0
    current_max = len(numbers) - 1
    current_guess = (current_min + current_max) // 2

    while current_min <= current_max:
        if numbers[current_guess] == target:
            return True
        elif numbers[current_guess] < target:
            current_min = current_guess + 1
        else:
            current_max = current_guess - 1
        current_guess = (current_max + current_min) // 2
    return False


result = is_exist_target_number_binary(finding_target, finding_numbers)
print(result)
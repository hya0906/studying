finding_target = 14
finding_numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

#내가 한 것-답은 나오나 이상함. 고쳐야 함
# def is_existing_target_number_binary(target, array):
#     if array[len(array)//2] > finding_target:
#         return is_existing_target_number_binary(target, array[0: len(array)//2])
#     elif array[len(array)//2] < finding_target:
#         return is_existing_target_number_binary(target, array[len(array) // 2:])
#     elif array[len(array)//2] == finding_target:
#         return True
#     return False

#강의해답
#순차적인 코드는 시간복잡도 O(N)
#이진탐색 시간복잡도 O(logN)
def is_existing_target_number_binary(target, array):
    current_min = 0
    current_max = len(array) - 1
    current_guess = (current_min + current_max) // 2

    while current_min <= current_max:
        if array[current_guess]  == target:
            return True
        elif array[current_guess] < target:
            current_min = current_guess + 1
        else:
            current_max = current_guess - 1
        current_guess = (current_max + current_min) // 2
    return False


result = is_existing_target_number_binary(finding_target, finding_numbers)
print(result)
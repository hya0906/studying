input = [3, 5, 6, 1, 2, 4]


def is_number_exist(number, array):
    for i in input: #array의 길이만큼 아래 연산 실행
        if i == number: #비교연산 1번만큼 실행
            return True #N * 1 = N
    return False


result = is_number_exist(3, input)
print(result)
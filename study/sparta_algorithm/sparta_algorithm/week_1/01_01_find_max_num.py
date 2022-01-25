input = [3, 5, 6, 1, 2, 4]

##내가한것
# def find_max_num(array):
#     max = 0
#     for i in array:
#         if max < i:
#             max = i
#     return max

##1번째 풀이 for else문
def find_max_num(array):
    for num in array:
        for compare_num in array:
            if num < compare_num:
                break
        else:
            return num
    return max

result = find_max_num(input)
print(result)
array = [5, 3, 2, 1, 6, 8, 7, 4]
# 5 3 -> [3,5]
# 2 1 -> [1,2]
# 6 8 -> [6,8]
# 7 4 -> [4,7]
#[3,5][1,2]->[1,2,3,5]
#[6,8][4,7]->[4,6,7,8] 길이 N/2를 2개를 비교하며 합침 N/2 * 2 = N
#[1,2,3,5][4,6,7,8]->[1,2,3,4,5,6,7,8] 길이 n
#병합단계는 모든 단계에서 N만큼 걸림
#k단계 -1개가 되는 지점
#N/2^k = 1 -> k = log2N
#즉 k단계 만큼 반복하는데 각각 단계는 O(N) 시간복잡도를 가진다.
#즉 log2N*O(N) = O(NlogN)

def merge_sort(array):
    if len(array) <= 1:
        return array
    mid = len(array) // 2
    left_array = merge_sort(array[:mid])
    right_array = merge_sort(array[mid:])
    return merge(left_array, right_array)


def merge(array1, array2):
    #len(array1) + len(array2) = O(N)
    result = []
    array1_index = 0
    array2_index = 0
    while array1_index < len(array1) and array2_index < len(array2):
        if array1[array1_index] < array2[array2_index]:
            result.append(array1[array1_index])
            array1_index += 1
        else:
            result.append(array2[array2_index])
            array2_index += 1

    if array1_index == len(array1):
        while array2_index < len(array2):
            result.append(array2[array2_index])
            array2_index += 1

    if array2_index == len(array2):
        while array1_index < len(array1):
            result.append(array1[array1_index])
            array1_index += 1

    return result


print(merge_sort(array))  # [1, 2, 3, 4, 5, 6, 7, 8] 가 되어야 합니다!
input = [4, 6, 2, 9, 1]


def bubble_sort(array): # O(N^2)
    for j in range(len(array)-1): #n의 길이
        n = len(array)
        for i in range(n -1 - j): #n의 길이
            if array[i] > array[i + 1]:
                array[i], array[i + 1] = array[i + 1], array[i]
    return array


bubble_sort(input)
print(input)  # [1, 2, 4, 6, 9] 가 되어야 합니다!
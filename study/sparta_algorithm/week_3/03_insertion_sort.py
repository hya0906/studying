input = [4, 6, 2, 9, 1]

#선택정렬 - 거의 정렬된 배열을 입력했을 경우 다른 것에 비해 조금 이득을 얻을 수 있다.
#버블정렬, 선택정렬 - O(N^2)

def insertion_sort(array):
    n = len(array)
    for i in range(1, n):
        for j in range(i):
            if array[i -j -1] > array[i - j]:
                array[i-j-1], array[i-j] = array[i-j], array[i-j-1]
            else:     ###
                break ###버블정렬과 선택정렬의 다른점
    return array


insertion_sort(input)
print(input) # [1, 2, 4, 6, 9] 가 되어야 합니다!
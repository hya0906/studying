#알파벳 빈도수 찾기
def find_alphabet_occurrence_array(string):
    alphabet_occurrence_array = [0] * 26
    for i in string:
        if i.isalpha():
            array_index = ord(i)-ord('a')
            alphabet_occurrence_array[array_index] += 1
        else:
            continue
    return alphabet_occurrence_array


print(find_alphabet_occurrence_array("hello my name is sparta"))
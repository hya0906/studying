input = "hello my name is sparta"

#두번째 풀이
def find_max_occurred_alphabet(string):
    alphabet_occurrence_array = [0] * 26
    for i in string:
        if i.isalpha():
            array_index = ord(i) - ord('a')
            alphabet_occurrence_array[array_index] += 1
        else:
            continue
    max_occurrence = 0
    max_alphabet_index = 0
    for index in range(len(alphabet_occurrence_array)):
        #index 0 -> alphabet_occurrence 3
        alphabet_occurrence = alphabet_occurrence_array[index]
        if alphabet_occurrence > max_occurrence:
            max_alphabet_index = index
            max_occurrence = alphabet_occurrence
    print(max_alphabet_index)
    return chr(max_alphabet_index + ord('a'))

result = find_max_occurred_alphabet(input)
print(result)
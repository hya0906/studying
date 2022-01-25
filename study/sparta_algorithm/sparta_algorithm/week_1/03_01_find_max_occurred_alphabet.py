input = "hello my name is sparta"

#내가 한 것
# def find_max_occurred_alphabet(string):
#     alphabet_occurrence_array = [0] * 26
#     for i in string:
#         if i.isalpha():
#             array_index = ord(i) - ord('a')
#             alphabet_occurrence_array[array_index] += 1
#         else:
#             continue
#     return chr(alphabet_occurrence_array.index(max(alphabet_occurrence_array)) + ord('a'))

#1번째풀이
def find_max_occurred_alphabet(string):
    alphabet_array = ["a", "b", "c","d","e","f","g","h","i","j","k","l","m",
                      "n","o","p","q","r","s","t","u","v","w","x","y","z"]

    max_occurrence = 0
    max_alphabet = alphabet_array[0]

    for alphabet in alphabet_array:
        occurrence = 0
        for char in string:
            if char == alphabet:
                occurrence += 1

        if occurrence > max_occurrence:
            max_occurrence = occurrence
            max_alphabet = alphabet

    return max_alphabet


result = find_max_occurred_alphabet(input)
print(result)
all_students = ["나연", "정연", "모모", "사나", "지효", "미나", "다현", "채영", "쯔위"]
present_students = ["정연", "모모", "채영", "쯔위", "사나", "나연", "미나", "다현"]

#O(N^2) - 비효율적임
#for all_student in all_students:
#    for present_students in present_students:

#O(N)-공간 많이 사용-공간 복잡도도 O(N)
#해쉬테이블은 시간은 최소화 시킬 수 있지만 공간을 대신 사용하는 구조이다.
def get_absent_student(all_array, present_array):
    student_dict = {}
    for key in all_array: #학생들 이름 추가 O(N)
        student_dict[key] = True

    for key in present_array: #출석한 학생 제거 O(N)
        del student_dict[key]

    for key in student_dict.keys(): #결석한 학생 1명 남음
        return key
    

print(get_absent_student(all_students, present_students))
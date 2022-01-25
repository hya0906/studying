class Node:
    def __init__(self, data):
        self.data = data
        self.next = None


class LinkedList:
    def __init__(self, value):
        self.head = Node(value)

    def append(self, value):
        cur = self.head
        while cur.next is not None:
            cur = cur.next
        cur.next = Node(value)

#링크드리스트 -> 끝을 알 수 없음
#1방법-전체의 길이를 알아낸 다음에 알아낸 길이 만큼만 -k만큼 하면 뒤에서k만큼 이동가능
    #->1.linkedlist길이 전부 알아내기 O(N) 2.그 길이에서 k만큼 뺀 길이 만큼 이동 O(N-K)

#2방법
    #1.노드를 두개 잡는다
    #2.한 노드를 다른 노드보다 k만큼 떨어지게 한다.
    #3.그리고 계속 한 칸씩 같이 이동한다.
    #4.만약 더 빠른 노드가 끝에 도달했다면
    #느린 노드는 끝에서 k만큼 떨어진 노드가 되므로 바로 반환하자 O(N)빠른노드 O(N-k)느린노드

    #->시간복잡도의 연산량이 같음
    def get_kth_node_from_last(self, k):
        slow = self.head
        fast = self.head

        for i in range(k):
            fast = fast.next #k만큼 앞으로 가도록 한다.

        while fast is not None: #끝까지 이동
            fast = fast.next
            slow = slow.next
            
        return slow

    # def get_kth_node_from_last(self, k):
    #     length = 1  # 시작 노드의 길이를 세기 위해 1부터 시작합니다
    #     cur = self.head
    #
    #     while cur.next is not None:  # cur을 끝까지 이동시킴
    #         cur = cur.next
    #         length += 1
    #     end_length = length - k  # 끝까지 이동시킨 후 k만큼 떨어진 곳
    #     cur = self.head
    #     for i in range(end_length):
    #         cur = cur.next
    #     return cur


linked_list = LinkedList(6)
linked_list.append(7)
linked_list.append(8)

print(linked_list.get_kth_node_from_last(2).data)  # 7이 나와야 합니다!
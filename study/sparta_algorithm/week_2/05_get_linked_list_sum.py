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


def get_linked_list_sum(linked_list_1, linked_list_2):
    sum_1 = _get_linked_list_sum(linked_list_1)
    sum_2 = _get_linked_list_sum(linked_list_2)
    # sum_1 = 0
    # head_1 = linked_list_1.head
    # while head_1 is not None:
    #     sum_1 = sum_1 * 10 + head_1.data
    #     head_1 = head_1.next
    #
    # sum_2 = 0
    # head_2 = linked_list_2.head
    # while head_2 is not None:
    #     sum_2 = sum_2 * 10 + head_2.data
    #     head_2 = head_2.next

    return sum_1 + sum_2

def _get_linked_list_sum(linked_list):
    linked_list_sum = 0
    head = linked_list.head
    while head is not None:
        linked_list_sum = linked_list_sum * 10 + head.data
        head = head.next
    return linked_list_sum

#[6] -> [7] -> [8]
linked_list_1 = LinkedList(6)
linked_list_1.append(7)
linked_list_1.append(8)

#[3] -> [5] -> [4]
linked_list_2 = LinkedList(3)
linked_list_2.append(5)
linked_list_2.append(4)

print(get_linked_list_sum(linked_list_1, linked_list_2))
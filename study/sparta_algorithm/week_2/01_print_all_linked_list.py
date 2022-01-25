#[3]->[4]
#data, next
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

node = Node(3)
first_node = Node(4)
node.next = first_node
print(node.next.data)
print(node.data)

#헤드노드만 가지고 있으면 됨
class LinkedList:
    def __init__(self, data):
        self.head = Node(data)

    def append(self, data):
        if self.head is None: #head가 없는 경우
            self.head = Node(data)
            return

        #self.head.next = Node(data)
        cur = self.head
        while cur.next is not None: #head가 있는 경우
            cur = cur.next  #cur이 맨 끝으로 이동할 수 있도록 함
        cur.next = Node(data)

    def print_all(self):
        print("hihihi")
        cur = self.head
        while cur is not None:
            print(cur.data)
            cur = cur.next
#                    head.next = Node(new)
#[3] -> [4] -> [5] -> [6] -> [new]
linked_list = LinkedList(3)
#print(linked_list.head.data)
#print(linked_list.head.next)

linked_list.append(4)
linked_list.append(5)
linked_list.print_all()
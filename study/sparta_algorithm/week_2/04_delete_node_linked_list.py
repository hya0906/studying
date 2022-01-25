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

    def print_all(self):
        cur = self.head
        while cur is not None:
            print(cur.data)
            cur = cur.next

    #내가 한 것
    # def get_node(self, index):
    #     cur = self.head
    #     for i in range(index):
    #         cur = cur.next
    #     return cur.data

    #강의해답
    def get_node(self, index):
        node = self.head
        count = 0
        while count < index:
            node = node.next
            count += 1
        return node

#     index   next_node
#     ['자갈'] ["밀가루"]->["우편"]
#              new_node
#            ->["흑연"] ->
# index.next - new_node
# new_node.next = next_node
    
    #내가 한것-못함
    # def add_node(self, index, value):
    #     cur = self.head
    #     for i in range(index):
    #         cur = cur.next
    #     cur.next = Node(value)
    #     cur.next =
    #     return

    def add_node(self, index, value):
        new_node = Node(value)  # [6]
        if index == 0:
            new_node.next = self.head #[6] -> [5]
            self.head = new_node #head -> [6] -> [5]->....
            return

        node = self.get_node(index - 1) #[5]
        next_node = node.next #[12]
        node.next = new_node #[5] - >[6]
        new_node.next = next_node #[6] -> [12]
        return

    def delete_node(self, index): #index번째 노드를 제거
        if index == 0:
            self.head = self.head.next
            return
        node = self.get_node(index-1)
        node.next = node.next.next
        return

# linked_list = LinkedList(5)
# linked_list.append(12)
# linked_list.append(8)
# # [5] -> [12] -> [8]
#
# # [5] -> [6] -> [12] -> [8]
# linked_list.add_node(1,6)
# #print(linked_list.get_node(1).data) # -> 5를 들고 있는 노드를 반환해야 합니다!
# linked_list.print_all()

#[5] -> [12]
linked_list = LinkedList(5)
linked_list.append(12)
linked_list.add_node(0,3)
linked_list.print_all()
linked_list.delete_node(0)
linked_list.print_all()
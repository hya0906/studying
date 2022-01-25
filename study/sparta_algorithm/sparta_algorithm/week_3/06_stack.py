class Node:
    def __init__(self, data):
        self.data = data
        self.next = None


class Stack:
    def __init__(self):
        self.head = None

    #(head)
    # [4]
    #(head)
    # [3] -> [4]
    def push(self, value):
        new_head = Node(value)
        new_head.next = self.head
        self.head = new_head

    #head값 없어져야하고 교체해야함.
    #[3]을 밖으로 내보내고 [4]가 head가 됨
    # pop 기능 구현
    def pop(self):
        if self.is_empty():
            return "Stack is empty"
        delete_head = self.head
        self.head = self.head.next
        return delete_head

    def peek(self):
        if self.is_empty():
            return "Stack is empty"
        return self.head.data

    # is_empty 기능 구현
    def is_empty(self):
        return self.head is None #비어있으면 true 있으면 false

stack = Stack()
stack.push(3)
print(stack.peek())
stack.push(4)
print(stack.peek())
print(stack.pop().data)
print(stack.peek())
print(stack.is_empty())
print(stack.pop().data)
print(stack.is_empty())
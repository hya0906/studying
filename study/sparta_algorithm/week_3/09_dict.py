class Dict:
    def __init__(self):
        self.items = [None] * 8

    # key -> ksdfksdf8 -> self.items[7] = "test"
    # key -> ksdfksdfk -> self.items[7] = "test"
    #인덱스가 같게 되면 덮어씌어짐->충돌->해결하기 위해 링크드리스트 사용
    # (키와 내용 같이 저장필요)
    # key -> ksdfksdfk -> self.items[7] = [("ksdfksdf8","test")] -> [("ksdfksdfk","test33")]
    def put(self, key, value):
        index = hash(key) % len(self.items)
        self.items[index] = value

    def get(self, key):
        index = hash(key) % len(self.items)
        return self.items[index]

my_dict = Dict()
my_dict.put("test", 3)
print(my_dict.get("test"))
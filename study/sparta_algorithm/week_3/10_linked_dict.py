#해쉬함수-상수시간이 걸리기 때문에 데이터를 찾고 추가하는 데에
# 상수시간만에 처리할 수 있다.
class LinkedTuple:
    def __init__(self):
        self.items = list()

    #[("ksdfksdf8","test")] -> [("ksdfksdfk","test33")]
    def add(self, key, value):
        self.items.append((key, value))

    def get(self, key):
        # [("ksdfksdf8","test"), ("ksdfksdfk","test33")]
        for k, v in self.items:
            if key == k:
                return v

class LinkedDict:
    def __init__(self):
        self.items = []
        for i in range(8):
            self.items.append(LinkedTuple())

    def put(self, key, value):
        index = hash(key) % len(self.itmes)
        #self.items[index] = value 기존구조 -여기에서는 linkedtuple이 인덱스에 존재
        self.items[index].add(key, value)

    def get(self, key):
        index = hash(key) % len(self.items)
        #LinkedTuple
        #[(key1, value1), (key, value)]
        return self.items[index].get(key)
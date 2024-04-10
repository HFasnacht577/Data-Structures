class Queue:

    def __init__(self):
        self.data = []

    def enqueue(self,item):
        self.data.append(item)

    def dequeue(self):
        return self.data.pop(0)

    def isempty(self):
        return self.data == []
    def size(self):
        return len(self.data)
    def __str__(self):
        return str(self.data)
    def __repr__(self):
        return self.__str__()
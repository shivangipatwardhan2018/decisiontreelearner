import heapq

#We can use this is a custom DataNode item type
class PriorityQueue:
    def __init__(self):
        self.queue = []
        self.index = 0

    def push(self, item, priority):
        heapq.heappush(self.queue, (-priority, self.index, item))
        self.index += 1

    def pop(self):
        return heapq.heappop(self.queue)[-1]

    def peekInformationGainDelta(self):
        element = heapq.heappop(self.queue)[-1]
        value = element.informationgain
        heapq.heappush(value)
        self.index += 1
        return value

    def isEmpty(self):
        def isEmpty(self):
            return self.index == 0

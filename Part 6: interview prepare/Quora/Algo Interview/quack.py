# Quack(int initialSize)
#     Initializes a quack with the given initial capacity.

# void push(T newElement)
#     Pushes a new element onto the top of the quack (stack behavior).

# T pop()
#     Removes and returns the topmost element from the quack (stack behavior).

# void enqueue(T newElement)
#     Adds a new element to the end of the quack (queue behavior).

# T dequeue()
#     Removes and returns the earliest inserted element from the quack (queue behavior).

# int currentSize()
#     Returns the current number of elements in the quack.

# NOTE: 
# 1. cannot use resizeable array 
# arr = [] ==> arr.append(xxx)
# 2. stack: last in first out
# 3. queue: first in first out

# push: [a, b, c] ==> [a, b, c, d]
# pop: [a, b, c, d] ==> [a, b, c] get thing from the right side
# enqueue: [a, b] ==> [a, b, c]
# deque: [a, b, c] ==> [b, c] get thing from the left side

class Quack:
    def __init__(self, size):
        self.capacity = size
        self.arr = [None]*self.capacity
        self.front = 0 # head of queue
        self.rear = 0 # next insert position
        self.length = 0 # current size
    
    # stack: push at rear
    def push(self, newElement):
        if self.length == self.currentSize:
            raise Exception("Quack is full")
        self.arr[self.rear] = newElement
        self.rear = (self.rear+1)%self.capacity
        self.length+=1
    
    # queue: enter from the right side
    def enqueue(self, newElement):
        self.push(newElement)
    
    # stack: pop from the right side
    def pop(self):
        if self.length==0:
            raise Exception("Quack is empty")
        self.rear = (self.rear-1+self.capacity)%self.capacity
        value = self.arr[self.rear]
        self.arr[self.rear] = None
        self.length-=1
        return value
    
    # queue: pop from the left side
    def dequeue(self):
        if self.length == 0:
            raise Exception("Quack is empty")
        value = self.arr[self.front]
        self.arr[self.front] = None
        self.front = (self.front+1)%self.capacity
        self.length-=1
        return value
    
    def currentSize(self):
        return self.length
    
    def show(self):
        res = []
        for i in range(len(self.arr)):
            res.append(self.arr[i])
        print(res)
        return res

q= Quack(10)
q.push('A')
q.push('B')
q.push('C')
q.show()

q.enqueue('D')
q.show()

q.pop()
q.show()

q.dequeue()
q.show()
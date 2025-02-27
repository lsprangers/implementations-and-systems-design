# 232

# [1, 2, 3, 4]
# pop should return 1
# other.push(first.pop()) --> [4, 3, 2, 1]
# other.pop() = 1
# first.push(other.pop()) --> [2, 3, 4]

class MyQueue:

    def __init__(self):
        self.new = []
        self.helper = []

    def push(self, x: int) -> None:
        self.new.append(x)
        

    def pop(self) -> int:
        # O(n) time and space
        while len(self.new) > 0:
            self.helper.append(self.new.pop())
        
        to_return = self.helper.pop()

        # O(n) time and space
        while len(self.helper) > 0:
            self.new.append(self.helper.pop())       
        
        return(to_return)

    def peek(self) -> int:
        # O(n) time and space
        while len(self.new) > 0:
            self.helper.append(self.new.pop())
        
        to_return = self.helper.pop()
        self.helper.append(to_return)

        # O(n) time and space
        while len(self.helper) > 0:
            self.new.append(self.helper.pop())       
        
        return(to_return)

    def empty(self) -> bool:
        return(len(self.new) == 0)
        


# Your MyQueue object will be instantiated and called as such:
# obj = MyQueue()
# obj.push(x)
# param_2 = obj.pop()
# param_3 = obj.peek()
# param_4 = obj.empty()
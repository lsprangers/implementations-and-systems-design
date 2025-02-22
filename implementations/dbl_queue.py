import sys
from dataclasses import dataclass
from typing import Optional 

@dataclass
class LinkedNode:
    val: int = 0
    next: 'LinkedNode' = None
    prev: 'LinkedNode' = None
    

class doubleEndQueue():
    def __init__(self):
        self.head = LinkedNode(sys.maxsize, None, None)
        self.tail = LinkedNode(sys.maxsize, None, None)
        self.head.next = self.tail
        self.tail.prev = self.head
        
        # for random helpers
        self.size = 0
    
    def pushleft(self, node: LinkedNode) -> None:
        curr_head = self.head.next
        self.head.next = node
        node.prev = self.head
        node.next = curr_head
        curr_head.prev = node

        self.size += 1
        return
    
    def pushright(self, node: LinkedNode) -> None:
        curr_tail = self.tail.prev
        curr_tail.next = node 
        node.prev = curr_tail
        node.next = self.tail
        self.tail.prev = node

        self.size += 1
        return
    
    def popleft(self) -> LinkedNode:
        if self.size == 0:
            raise Exception("No Nodes!")

        curr_head = self.head.next
        self.head.next = curr_head.next
        curr_head.next.prev = self.head
        curr_head.next = None
        curr_head.prev = None

        self.size -= 1
        return(curr_head)
    
    def popright(self) -> LinkedNode:
        if self.size == 0:
            raise Exception("No Nodes!")

        curr_tail = self.tail.prev
        self.tail.prev = curr_tail.prev
        curr_tail.prev.next = self.tail
        curr_tail.next = None
        curr_tail.prev = None

        self.size -= 1
        return(curr_tail)

    def breakSentinels(self):
        curr_head = self.head.next
        curr_tail = self.tail.prev
        
        # break sentinels
        curr_head.prev = None
        self.head.next = None
        curr_tail.next = None
        self.tail.prev = None 
        return(curr_head, curr_tail)
         
    
    def sort(self, head: Optional[LinkedNode]):
        # psycho to make someone do this in an interview
        if not head or not head.next:
            return(head)
        
        mid = self.getMidAndCut(head)
        left = self.sort(head)
        right = self.sort(mid)
        
        return(self.merge(left, right))
        
    
    def getMidAndCut(self, head: Optional[LinkedNode]):
        if not head:
            return
        
        slow = head
        fast = head
        while fast.next and fast.next.next:
            slow = slow.next
            fast = fast.next.next
        
        mid = slow.next
        slow.next = None
        if mid:
            mid.prev = None
        
        return mid
        
        
        
    
    def merge(self, left: LinkedNode, right: LinkedNode) -> LinkedNode:
        if not left and not right:
            return
        # if anyone makes me do this in an interview I'll scream
        tmp_head = LinkedNode(-1)
        curr = tmp_head

        while left and right:
            if left.val <= right.val:
                curr.next = left
                left.prev = curr
                left = left.next
            else:
                curr.next = right
                right.prev = curr
                right = right.next
            curr = curr.next

        # Attach the remaining nodes, if any
        if left:
            curr.next = left
            left.prev = curr
        elif right:
            curr.next = right
            right.prev = curr

        # The merged list starts from tmp_head.next
        if tmp_head.next:
            tmp_head.next.prev = None

        return tmp_head.next
            

    def to_array(self):
        values = []
        current = self.head.next
        while current != self.tail:
            values.append(str(current.val))
            current = current.next
            
        return(values)
                    
    def __str__(self) -> str:
        return " <-> ".join(self.to_array())  
    


if __name__ == "__main__":
    deq = doubleEndQueue()
    # assert(deq.popleft() == Exception("No Nodes!"))

    one = LinkedNode(1)
    two = LinkedNode(2)
    three = LinkedNode(3)
    
    deq.pushleft(one)
    deq.pushleft(two)
    assert(deq.popleft() == two)
    
    deq.pushright(two)
    deq.pushright(three)    
    
    print(deq)

    assert(deq.popleft() == one)
    deq.pushright(one)
    assert(deq.popright() == one)
    deq.pushleft(one)
        
        
    eight = LinkedNode(8)
    nine = LinkedNode(9)
    ten = LinkedNode(10)
    deq.pushleft(ten)
    deq.pushleft(nine)   
    deq.pushleft(eight)   
    
    # sentinels and sort are meh 
    
    # print("pre sort", deq)
    
    # # head, tail = deq.breakSentinels()
    # deq.sort(deq.head.next)
    
    # print("post sort", deq)
    
    # assert(deq.to_array == [1, 2, 3, 8, 9, 10])     
    
    # print(deq)  
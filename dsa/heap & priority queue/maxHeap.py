
class maxHeap:
    def __init__(self, heap_list = []):
        self.heap_list = heap_list
    
    def __len__(self):
        return(len(self.heap_list))

    def swap_values(self, index1, index2):
        self.heap_list[index1], self.heap_list[index2] = self.heap_list[index2], self.heap_list[index1]

    def heapify(self):
        """Heapify a tree, it will start at root unless specified
        """
        # n is first non-leaf node, and we basically will loop backwards from it up to root.
        # Inside of _heapify_node we will recurse down through the subtree, so if we do end
        #   up swapping something on this node we'll be sure this subtree is heapified.
        # Meaning that as we go from first non-leaf node up to root, we know the subtree is 
        #   correct
        n = len(self.heap_list)
        for node_idx in range( (n // 2) - 1, -1, -1):
            self._heapify_node(node_idx)
        
    def _heapify_node(self, node_idx) -> None:
        """_heapify_node is called for a specific node to ensure the heap property
        is true for it and its children

        Args:
            node_idx (int): Index of node you want to heapify for it's children. 
        """
        largest_idx = node_idx
        left_idx = (2*node_idx) + 1
        right_idx = (2*node_idx) + 2
        
        if left_idx < len(self.heap_list) and self.heap_list[left_idx] > self.heap_list[largest_idx]:
            largest_idx = left_idx
        if right_idx < len(self.heap_list) and self.heap_list[right_idx] > self.heap_list[largest_idx]:
            largest_idx = right_idx            
        
        if largest_idx != node_idx:
            self.swap_values(node_idx, largest_idx)
            self._heapify_node(largest_idx)

    def insert(self, newNum) -> None:
        """Inserts a number at the end of maxHeap, and then swaps values
        with parent until heap property is met

        Args:
            newNum (int): New number to add into heap
        """
        self.heap_list.append(newNum)
        curr_idx = len(self.heap_list) - 1
        while curr_idx > 0:
            parent_idx = (curr_idx - 1) // 2
            if self.heap_list[curr_idx] > self.heap_list[parent_idx]:
                self.swap_values(curr_idx, parent_idx)
                curr_idx = parent_idx
            else:
                break   

    def deleteNode(self, num):
        """Delete a node from the heap.
        We will just move it to the end, pop it off, and then
        run heapify from where the num used to be

        Args:
            num (_type_): _description_
        """
        size = len(self.heap_list)
        
        # find the index of the num
        num_idx = self.heap_list.index(num)

        # Swap with the last element
        self.swap_values(num_idx, -1)
        self.heap_list.pop()  # Remove the last element which is now the number to be deleted

        # Only run heapify if the deleted node was not the last node
        if num_idx < size:
            self._heapify_node(num_idx)       
        

heap = maxHeap()
heap.heap_list = [3, 9, 2, 1, 4, 5]
heap.heapify()
print(heap.heap_list)
heap.insert(100)
heap.insert(50)
heap.insert(60)
print(heap.heap_list)
heap.deleteNode(50)
print(heap.heap_list)
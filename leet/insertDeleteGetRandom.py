# 380

class RandomizedSet:

    def __init__(self):
        # O(1) find, insert, remove
        self.freqs = {}

        # O(1) random access
        # getting last item just requires keeping
        #   track of len and len-1
        self.val_list = []

        # Helper
        self.n_items = 0

    def found(self, val: int) -> bool:
        if val in self.freqs:
            return(True)
        return(False)

    def insert(self, val: int) -> bool:
        if(self.found(val)):
            return(False)
        
        self.freqs[val] = self.n_items
        self.val_list.append(val)
        self.n_items += 1
        return(True)

    def remove(self, val: int) -> bool:
        if not self.found(val):
            return(False)
        
        last_val = self.val_list[self.n_items - 1]
        remove_val_idx = self.freqs[val]
        self.val_list[remove_val_idx] = last_val
        self.freqs[last_val] = remove_val_idx
        self.val_list.pop()
        self.n_items -= 1
        del self.freqs[val]
        return(True)



    def getRandom(self) -> int:
        return(
            choice(self.val_list)
        )


# Your RandomizedSet object will be instantiated and called as such:
# obj = RandomizedSet()
# param_1 = obj.insert(val)
# param_2 = obj.remove(val)
# param_3 = obj.getRandom()
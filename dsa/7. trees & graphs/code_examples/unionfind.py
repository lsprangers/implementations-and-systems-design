class UnionFind:
    def __init__(self, size):
        self.root = [i for i in range(size)]
        self.rank = [1] * size
        return
    
    def find(self, pt):
        if(pt == self.root[pt]):
            return(pt)
        self.root[pt] = self.find(self.root[pt]) 
        
        return(self.root[pt])
    
    def union(self, pt1, pt2) -> bool:
        root1 = self.find(pt1)
        root2 = self.find(pt2)
        if(root1 != root2):
            if(self.rank[root1] > self.rank[root2]):
                # if second point is flatter, merge it into 1
                self.root[root2] = root1
            elif(self.rank[root1] < self.rank[root2]):
                # if first point is flatter, merge it into 2
                self.root[root1] = root2
            else:
                # merge second under first, and update it's rank
                self.root[root2] = root1
                self.rank[root1] += 1
            
            return True
        return False
                
    
    def connected(self, pt1, pt2):
        return(self.find(pt1) == self.find(pt2))    


# Example usage:
uf = UnionFind(10)
uf.union(1, 2)
uf.union(2, 3)
print(uf.connected(1, 3))  # Output: True
print(uf.connected(1, 4))  # Output: False
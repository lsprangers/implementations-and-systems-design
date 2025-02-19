# 1570
class SparseVector:
    def __init__(self, nums: List[int]):
        self.idx_to_num = {}
        for idx, num in enumerate(nums):
            if num > 0:
                self.idx_to_num[idx] = num

    # Return the dotProduct of two sparse vectors
    def dotProduct(self, vec: 'SparseVector') -> int:
        resp = 0
        for idx, num in vec.idx_to_num.items():
            if idx in self.idx_to_num:
                resp += self.idx_to_num[idx] * vec.idx_to_num[idx]
        return(resp)

# Your SparseVector object will be instantiated and called as such:
# v1 = SparseVector(nums1)
# v2 = SparseVector(nums2)
# ans = v1.dotProduct(v2)
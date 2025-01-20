class Solution:

    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        self.ara = nums1 + nums2
        self.ara.sort()
        if len(self.ara) % 2 == 0:
            return(self.ara[(len(self.ara)//2) - 1] + self.ara[(len(self.ara)//2)]) / 2
        else:
            return(self.ara[(len(self.ara)//2)])
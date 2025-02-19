#  162
class Solution:
    def findPeakElement(self, nums: List[int]) -> int:
        if len(nums) < 2:
            return(0)
            
        for idx, num in enumerate(nums):
            if idx == 0:
                if num > nums[idx + 1]:
                    return(idx)
            elif idx == len(nums) - 1:
                if num > nums[idx - 1]:
                    return(idx)
            
            else:
                if num > nums[idx - 1] and num > nums[idx + 1]:
                    return(idx)
        
        return(-1)
# 2874
class Solution:
    def maximumTripletValue(self, nums: List[int]) -> int:
        n = len(nums)
        leftMax = [0] * n
        rightMax = [0] * n
        # THe only reason this is able to be O(n) is b/c of equation
        # If it wasn't dependent on largest separation * largest value, we'd be stuck
        #   at O(n^3)

        for i in range(1, n):
            # drag along leftMax, so this is O(1)
            leftMax[i] = max(leftMax[i - 1], nums[i - 1])
            rightMax[n - i - 1] = max(nums[n - i], rightMax[n - i])
        
        resp = 0
        for i in range(1, n-1):
            resp = max(resp, (leftMax[i] - nums[i]) * rightMax[i])
        return(resp)
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        max_so_far = -100000
        idx_so_far = 0
        idx_start = 0
        max_here = 0

        for idx, n in enumerate(nums):
            max_here += n
            if max_so_far < max_here:
                max_so_far = max_here
                idx_so_far = idx

            # Whenever the sub-array up to "here" is negative, it 
            #   can't contribute anything, so we move on
            #   and start new sub-array
            if max_here < 0:
                idx_start = idx + 1
                max_here = 0
        
            print(
                f"n: {n}",
                # f"idx_start: {idx_start}", 
                # f"idx_so_far: {idx_so_far}", 
                f"max_here: {max_here}", 
                f"max_so_far: {max_so_far}")
        return max_so_far
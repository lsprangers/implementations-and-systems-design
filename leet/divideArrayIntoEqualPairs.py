# 2206
class Solution:
    def divideArray(self, nums: List[int]) -> bool:
        n = len(nums) // 2

        # gets us the frequency of each number quickly
        freqs = Counter(nums)

        return(all([freq % 2 == 0 for freq in freqs.values()]))
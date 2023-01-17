# 926
# "011001"
class Solution:
    def minFlipsMonoIncr(self, s: str) -> int:
        ones, res = 0, 0
        
        for bitt in s:
            if bitt =='1': ones += 1
            elif ones:
                ones -= 1
                res += 1

        return res
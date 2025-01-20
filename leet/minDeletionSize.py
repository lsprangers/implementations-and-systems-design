# problem 945
from typing import List

class Solution:
    def minDeletionSize(self, strs: List[str]) -> int:
        countt = 0
        for j in range(len(strs[0])):
            for i in range(len(strs) - 1):
                if strs[i + 1][j] < strs[i][j]:
                    countt += 1
                    break
        
        return countt 
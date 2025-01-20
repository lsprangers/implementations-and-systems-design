# problem 134
from typing import List
class Solution:
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        if sum(gas) < sum(cost):
            return -1
        
        current = idx = 0
        for i in range(len(gas)):
            current += gas[i]-cost[i]
            if current < 0: current, idx = 0, i+1
        return idx 
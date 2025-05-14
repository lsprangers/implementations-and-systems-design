# 2094

class Solution:
    def findEvenNumbers(self, digits: List[int]) -> List[int]:
        # O(n)
        freqs = Counter(digits)

        resp = []

        # O(1)
        # Need to do the freqs - and + 1 to ensure we don't reuse a number throughout the creation process
        for n1 in range(0, 10):
            if freqs[n1] == 0 or n1 == 0:
                continue
            freqs[n1] -= 1
            for n2 in range(0, 10):
                if freqs[n2] == 0:
                    continue
                freqs[n2] -= 1
                for e in range(0, 10, 2):
                    if freqs[e] == 0:
                        continue
                    freqs[e] -= 1
                    
                    heapq.heappush(resp, (n1*100 + n2*10 + e))
                    freqs[e] += 1
                freqs[n2] += 1
            freqs[n1] += 1
                
                    
        return(resp)
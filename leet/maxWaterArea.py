# 11

class Solution:
    def maxArea(self, height: List[int]) -> int:
        start=0
        end=len(height)-1
        maxx=0

        while start < end:
            x = min(height[start], height[end])
            area = x * (end - start)
            maxx = max(maxx, area) 
            if height[start]<height[end]:
                start+=1
            else:
                end-=1        
        return maxx
                
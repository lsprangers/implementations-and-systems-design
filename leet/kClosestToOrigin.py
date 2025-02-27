# 973
class Solution:
    def mdistance(self, a, b):
        return(
            sqrt(
                pow((a[0] - b[0]), 2) +
                pow((a[1] - b[1]), 2)
            )
        )

    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        pq = []
        for point in points:
            # maxheap to keep lowest in - i.e. pop() returns largest point
            heapq.heappush(
                pq, (-self.mdistance(point, [0, 0]), point)
            )
            while len(pq) > k:
                curr = heapq.heappop(pq)
        
        return([pt for dist, pt in pq])
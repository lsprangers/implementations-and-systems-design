# 135

class Solution:
    def candy(self, ratings: List[int]) -> int:
        candies = [1] * len(ratings)

        # O(n logn) still, and generally always
        new_ratings = [(rating, idx) for idx, rating in enumerate(ratings)]
        heapq.heapify(new_ratings)

        # [1,3,4,5,2]
        # [1,2,3,4,1]
        # [(0, 1), (1, 0), (2, 2)]
        # heapq.heappop(new_ratings) = (0,1)
        
        # O(n * logn)
        # for idx, ratin in enumerate(ratings):
        #     heapq.heappush(new_ratings, (ratin, idx))
        
        # O(n)
        while(len(new_ratings) > 0):
            smallest = heapq.heappop(new_ratings)
            curr_rating = smallest[0]
            curr_idx = smallest[1]

            # O(1)
            largest_smallest_neighbor_idx = self.largest_neighbor_less_than_us(curr_idx, ratings, candies)
            if(largest_smallest_neighbor_idx == -1):
                continue

            if(curr_rating > ratings[largest_smallest_neighbor_idx]):
                candies[curr_idx] = candies[largest_smallest_neighbor_idx] + 1

        return(sum(candies))

    
    def largest_neighbor_less_than_us(self, curr_idx: int, ratings: List[int], candies: List[int]) -> int:
        valid_neighbors = []
        # -item for max heap

        # check left
        if curr_idx > 0:
            if ratings[curr_idx - 1] < ratings[curr_idx]:
                heapq.heappush(valid_neighbors, (-candies[curr_idx - 1], curr_idx - 1))
        
        # check right
        if curr_idx < len(ratings) - 1:
            if ratings[curr_idx + 1] < ratings[curr_idx]:
                heapq.heappush(valid_neighbors, (-candies[curr_idx + 1], curr_idx + 1))
        
        if(len(valid_neighbors) > 0):
            return(heapq.heappop(valid_neighbors)[1])

        return(-1)
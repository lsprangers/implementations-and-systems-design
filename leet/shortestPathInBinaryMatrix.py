# 1091

class Solution:
    def __init__(self):
        self.moves = [
            [-1, -1],
            [-1, 0],
            [-1, 1],
            [0, -1],
            [0, 1],
            [1, 0],
            [1, 1],
            [1, -1]
        ]
        self.seen = set()
        self.n_rows = 0
        self.n_cols = 0
        self.resp = float("inf")

    def shortestPathBinaryMatrix(self, grid: List[List[int]]) -> int:
        if grid[0][0] != 0:
            return(-1)
        
        self.n_rows = len(grid)
        self.n_cols = len(grid[0])
        que = deque([])
        que.appendleft([0, 0, 1])
        while que:
            curr_size = len(que)
            for i in range(curr_size):
                curr = que.pop()
                row = curr[0]
                col = curr[1]
                num_cells = curr[2]
                if row == self.n_rows - 1 and col == self.n_cols - 1:
                    self.resp = min(self.resp, num_cells)

                if (row, col) in self.seen:
                    continue

                
                self.seen.add((row, col))
                for move in self.moves:
                    new_row = row + move[0]
                    new_col = col + move[1]
                    if self.checkRowCol(new_row, new_col) and grid[new_row][new_col] == 0:
                        que.appendleft([new_row, new_col, num_cells + 1])
        
        return(self.resp if self.resp != float("inf") else -1)


    def checkRowCol(self, row, col):
        return(
            row > -1
            and row < self.n_rows
            and col > -1
            and col < self.n_cols
            and (row, col) not in self.seen
        )
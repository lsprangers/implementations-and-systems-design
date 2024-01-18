#include <vector>;
#include <queue>;
using namespace std;
// 1091

class Solution {
public:

    int n_rows;
    int n_cols;
    vector<vector<int>> this_grid;
    int shortestPathBinaryMatrix(vector<vector<int>>& grid) {
        // O(n^2) time and space to initialize all this
        this_grid = grid;
        vector<vector<int>> dirs = { {-1,-1}, {-1,0}, {-1,1}, {0,-1}, {0,1}, {1,-1}, {1,0}, {1,1} };
        queue<vector<int>> que;        
        n_rows = this_grid.size();
        n_cols = this_grid[0].size();
        vector<vector<bool>> seen(n_rows, vector<bool>(n_cols, false));

        if(n_rows == 0 || this_grid[0][0] == 1) {
            return(-1);
        }
        seen[0][0] = true;
        que.push({0, 0, 1});

        while(!que.empty()) {
            int q_size = que.size();
            for(int i = 0; i < que.size(); i++) {
                vector<int> curr = que.front();
                que.pop();
                int row = curr[0];
                int col = curr[1];
                int steps = curr[2];
                if(row == (n_rows - 1) && col == (n_cols - 1)) {
                    return(steps);
                }
                else {
                    for(vector<int> dir : dirs) {
                        int nextRow = row + dir[0];
                        int nextCol = col + dir[1];
                        if(is_valid(nextRow, nextCol) && !seen[nextRow][nextCol]) {
                            seen[nextRow][nextCol] = true;
                            que.push({nextRow, nextCol, steps + 1});
                        }
                    }
                }

            }
            
        }
        return(-1);
    }


    bool is_valid(int row, int col) {
        return(0 <= row && row < n_rows && 0 <= col && col < n_cols && this_grid[row][col] == 0);
    }
};
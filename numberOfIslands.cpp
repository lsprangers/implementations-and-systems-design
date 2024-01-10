#include <vector>;
using namespace std;
// 200

class Solution {
public:
    int n_rows;
    int n_cols;
    vector<vector<char>> grid;
    // up, right, down, left
    vector<vector<int>> directions = {{0, 1}, {1, 0}, {0,-1}, {-1, 0}};
    vector<vector<bool>> seen;
    int numIslands(vector<vector<char>>& grid) {
        this->grid = grid;
        n_rows = grid.size();
        n_cols = grid[0].size();
        seen = vector(n_rows, vector<bool>(n_cols, false));

        int ans = 0;
        for(int row = 0; row < n_rows; row++) {
            for(int col = 0; col < n_cols; col++) {
                if(grid[row][col] == '1' && !seen[row][col]) {
                    ans++;
                    seen[row][col] = true;
                    dfs(row, col);
                }
            }
        }
        
        return(ans);
    }

    void dfs(int row, int col) {
        for (vector<int>& direction: directions) {
            int nextRow = row + direction[0], nextCol = col + direction[1];
            if (valid(nextRow, nextCol) && !seen[nextRow][nextCol]) {
                seen[nextRow][nextCol] = true;
                dfs(nextRow, nextCol);
            }
        }
    }
    
    bool valid(int row, int col) {
        return 0 <= row && row < n_rows && 0 <= col && col < n_cols && grid[row][col] == '1';
    }

};
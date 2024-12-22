// 64
class Solution {
public:

    int minPathSum(vector<vector<int>>& grid) {
        int n_rows = grid.size() - 1;
        int n_cols = grid[0].size() - 1;
        vector<vector<int>> cache(n_rows + 1, vector<int>(n_cols + 1));

        for(int i = n_rows; i > -1; i--) {
            for(int j = n_cols; j > -1; j--) {
                if(i == n_rows && j < n_cols) {
                    cache[i][j] = grid[i][j] + cache[i][j + 1]; // bottom row, not last col
                }
                else if(i < n_rows && j == n_cols) {
                    cache[i][j] = grid[i][j] + cache[i + 1][j]; // last col, not bottom row
                }
                else if(i < n_rows && j < n_cols) {
                    cache[i][j] = grid[i][j] + min(
                        cache[i + 1][j],
                        cache[i][j + 1]
                    );
                }
                else {
                    cache[i][j] = grid[i][j];
                }
            }
        }
        return(cache[0][0]);
    }

};
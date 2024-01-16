#include <vector>;
#include <set>;
using namespace std;
// 695
class Solution {
public:
    int resp = 0;
    set<vector<int>> seen;
    vector<vector<int>> directions = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
    vector<vector<int>> our_grid;
    int n_rows = 0;
    int n_cols = 0;
    int maxAreaOfIsland(vector<vector<int>>& grid) {
        our_grid = grid;
        n_rows = our_grid.size();
        n_cols = our_grid[0].size();

        for(int row = 0; row < n_rows; row++) {
            for(int col = 0; col < n_cols; col++) {
                if(our_grid[row][col] == 1 && seen.find({row, col}) == seen.end()) {
                    resp = max(resp, dfs(row, col));
                }
            }
        }
        return(resp);
    }
    int dfs(int row, int col) {
        if(!is_valid(row, col)) {
            return(0);
        }
        seen.insert({row, col});
        return(
            1 + dfs(row + 1, col) + dfs(row - 1, col) + dfs(row, col + 1) + dfs(row, col - 1)
        );
    }

    bool is_valid(int row, int col) {
        return(0 <= row && row < n_rows && 0 <= col && col < n_cols && seen.find({row, col}) == seen.end() && our_grid[row][col] == 1);
    }
};
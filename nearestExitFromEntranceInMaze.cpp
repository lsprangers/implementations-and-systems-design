#include <vector>;
#include <queue>;
using namespace std;

//1926

class Solution {
public:
    int n_rows;
    int n_cols;
    vector<vector<char>> this_maze;
    int entrance_row;
    int entrance_col;

    int nearestExit(vector<vector<char>>& maze, vector<int>& entrance) {
        this_maze = maze;
        vector<vector<int>> dirs = { {-1,0}, {0,-1}, {0,1}, {1,0}};
        queue<vector<int>> que;        
        n_rows = this_maze.size();
        n_cols = this_maze[0].size();
        vector<vector<bool>> seen(n_rows, vector<bool>(n_cols, false));
        int resp = INT_MAX;

        if(n_rows == 0) {
            return(-1);
        }
        entrance_row = entrance[0];
        entrance_col = entrance[1];
        seen[entrance_row][entrance_col] = true;
        que.push({entrance_row, entrance_col, 0});

        while(!que.empty()) {
            int q_size = que.size();
            for(int i = 0; i < q_size; i++) {
                vector<int> curr = que.front();
                que.pop();
                int currRow = curr[0];
                int currCol = curr[1];
                int currSteps = curr[2];
                if(is_exit(currRow, currCol)) {
                    resp = min(resp, currSteps);
                }                
                
                for(vector<int> dir : dirs) {
                    int nextRow = currRow + dir[0];
                    int nextCol = currCol + dir[1];
                    if(is_valid(nextRow, nextCol) && !seen[nextRow][nextCol]) {
                        seen[nextRow][nextCol] = true;
                        que.push({nextRow, nextCol, currSteps + 1});
                    }
                }
            }
        }
        resp = (resp == INT_MAX) ? -1 : resp;
        return(resp);
    }

    bool is_exit(int row, int col) {
        return( (row == 0 || col == 0 || row == n_rows-1 || col == n_cols - 1) && this_maze[row][col] == '.' && (row != entrance_row || col != entrance_col));
    } 
    bool is_valid(int row, int col) {
        return(0 <= row && row < n_rows && 0 <= col && col < n_cols && this_maze[row][col] == '.');    
    }     
};
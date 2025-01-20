#include <vector>;
using namespace std;

//74

class Solution {
public:
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        int startRow = 0;
        int finalRow = matrix.size()-1;
        while(startRow <= finalRow) {
            int middleRowIdx = startRow + ((finalRow - startRow) / 2);
            vector<int> middleRow = matrix[middleRowIdx];
            if(middleRow[0] <= target && middleRow[middleRow.size() - 1] >= target) {
                return(bfs(middleRow, target));
            }
            else if(middleRow[0] > target) {
                finalRow = middleRowIdx - 1;
            }
            else {
                startRow = middleRowIdx + 1;
            }
        }
        return(false);
    }
    bool bfs(vector<int> matrixRow, int target) {
        int left = 0;
        int right = matrixRow.size() - 1;
        while(left <= right) {
            int middle = left + ((right - left) / 2);
            if(matrixRow[middle] == target) {
                return(true);
            }
            else if(matrixRow[middle] > target) {
                right = middle - 1;
            }
            else {
                left = middle + 1;
            }
        }
        return(false);
    }
};
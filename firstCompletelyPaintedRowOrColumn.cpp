// 2661
class Solution {
public:
    int firstCompleteIndex(vector<int>& arr, vector<vector<int>>& mat) {
        int n_rows = mat.size();
        int n_cols = mat[0].size();
        int col_size = n_rows;
        int row_size = n_cols;

        unordered_map<int, pair<int, int>> matrixIdx;
        for(int i = 0; i < n_rows; i++) {
            for(int j = 0; j < n_cols; j++) {
                matrixIdx[mat[i][j]] = pair(i, j);
            }
        }

        // freqs initialized to 0
        //  for any row, we hope to get to n_rows, and same for cols
        unordered_map<int, int> row_freqs;
        unordered_map<int, int> col_freqs;
        
        for(int k = 0; k < arr.size(); k++) {
            int num = arr[k];

            col_freqs[matrixIdx[num].second]++;
            if(col_freqs[matrixIdx[num].second] >= col_size) {
                return(k);
            }

            row_freqs[matrixIdx[num].first]++;
            if(row_freqs[matrixIdx[num].first] >= row_size) {
                return(k);
            }

        }
        return(-1);

    }
};
#include <vector>;
using namespace std;
//77
class Solution {
public:
    vector<vector<int>> combine(int n, int k) {
        vector<vector<int>> resp;
        vector<int> curr = {};
        backtrack(resp, curr, 1, n, k);
        return(resp);
    }
    void backtrack(vector<vector<int>>& resp, vector<int>& curr, int idx, int n, int k) {
        if(curr.size() == k) {
            resp.push_back(curr);
            return;
        }

        for(int i = idx; i < n + 1; i++) {
            curr.push_back(i);
            backtrack(resp, curr, i + 1, n, k);
            curr.pop_back();
        }
    }
};
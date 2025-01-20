#include <vector>;
using namespace std;
//78
class Solution {
public:

    vector<vector<int>> subsets(vector<int>& nums) {
        vector<vector<int>> resp;
        vector<int> curr = {};
        backtrack(curr, resp, nums, 0);
        return(resp);
    }
    void backtrack(vector<int>& curr, vector<vector<int>>& resp, vector<int>& nums, int idx) {
        if(idx > nums.size()) {
            return;
        }
        resp.push_back(curr);
        for(int i = idx; i < nums.size(); i++) {
            curr.push_back(nums[i]);
            backtrack(curr, resp, nums, i + 1);
            curr.pop_back();
        }
    }
};
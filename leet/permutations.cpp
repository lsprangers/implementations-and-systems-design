#include <vector>;
using namespace std;
// 46
class Solution {
public:
    vector<vector<int>> permute(vector<int>& nums) {
        vector<vector<int>> resp;
        vector<int> curr = {};
        backtrack(curr, resp, nums);
        return(resp);
    }
    void backtrack(vector<int>& curr, vector<vector<int>>& resp, vector<int>& nums) {
        if(curr.size() == nums.size()) {
            resp.push_back(curr);
            return;
        }
        for(int num : nums) {
            // if we didn't find it
            if(find(curr.begin(), curr.end(), num) == curr.end()) {
                // push back a number if we didn't find it
                // Example: let's say curr is [1]
                //  we will call backtrack, and skip over 1 in the for loop
                //  then we will see 2, and create [1, 2]
                //  and call backtrack again, which will skip 1 & 2 and
                //  create [1, 2, 3] - then next call to backtrack will 
                //  push it to resp, and return, and at that point we should
                //  pop off 3...then it will complete this for loop
                //  and curr will be [1, 2] and then backtrack call on
                //  stack will finish, and then we pop off 2, and we will
                //  be on the initial backtrack call for 1, and have just 
                //  popped off 2, and then it'll be [1, 3] and we'll call backtrack
                //  which will skip 1, not find 2, and create [1, 3, 2] and so on
                curr.push_back(num);
                backtrack(curr, resp, nums);
                curr.pop_back();
            }
        }
    }
};
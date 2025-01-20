
#include <vector>;
using namespace std;
// # 525 leetcode

class Solution {
public:
    int findMaxLength(vector<int>& nums) {
        unordered_map<int, int> umap;
        umap[0] = -1;

        int resp = 0;
        int curr = 0;

        for(int i = 0; i < nums.size(); i++) {
            // get current count at this index
            // inc / dec based on 1 / 0
            curr += (nums[i] == 1 ? 1 : -1);

            // If we've seen current count before, update resp
            if(umap.find(curr) != umap.end()) {
                // i - umap[curr] is current index - last seen index
                //  the index for 0 is -1 because at the start of the array
                // there is an equal number of 1's and 0's
                // So even if the count = 1 somewhere, and 7,000 entries later
                //  the count = 1 then we know our subarray with equal number
                //    of ones and 0's is 6,999
                // umap[0] = -1 is because equal number of 1's and 0's at start,
                //  but it doesn't mean our array has to start there
                resp = max(resp, i - umap.at(curr));
            }
            else {
                // otherwise we store the first value we've seen of this count
                umap[curr] = i;
            }
        }
    return(resp);
    }
};
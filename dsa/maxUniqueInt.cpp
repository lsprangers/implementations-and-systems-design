#include <vector>;
using namespace std;

class Solution {
public:
    int largestUniqueNumber(vector<int>& nums) {
        unordered_map<int, int> umap;
        for(int num : nums) {
            umap[num]++;
        }
        
        int resp = -1;
        for(auto [key, val] : umap) {
            if(val == 1) {
                resp = max(resp, key);
            }
        }
        return(resp);
    }
};
#include <vector>;
#include <algorithm>;
using namespace std;

// 2610
class Solution {
public:
    vector<vector<int>> findMatrix(vector<int>& nums) {
        unordered_map<int,int> umap;
        vector<vector<int>> resp = {{}};
        int maxVecs = 0;
        for( int n : nums) {
            if(umap.find(n) != umap.end()) {
                if (umap[n] == maxVecs) {
                    resp.push_back({}); // Create empty vectors for us to access later
                }
                maxVecs = max(umap[n] + 1, maxVecs);
                umap[n]++; // need to put this in another array
            }
            else {
                umap[n] = 0;
            }
            // cout << "n: " << n << " idx: " << umap[n] << "\n";
        }
        for (const auto & [ key, value ] : umap) {
            for(int i = 0; i <= value; i++) {
                resp[i].push_back(key);
            }
        }
        return(resp);
    }
};
#include <vector>;
using namespace std;

// 2260 leet
class Solution {
public:
    int minimumCardPickup(vector<int>& cards) {
        unordered_map<int, int> umap;
        int resp = INT_MAX;
        for(int i = 0; i < cards.size(); i++) {
            if(umap.find(cards[i]) != umap.end()) {
                resp = min(resp, i - umap.at(cards[i]));
            }
            umap[cards[i]] = i;
        }
        resp = (resp == INT_MAX ? -2 : resp);
        return(resp + 1);
    }
};
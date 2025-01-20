#include <vector>;
#include <iostream>;
#include <set>;
using namespace std;
// 771
class Solution {
public:
    int numJewelsInStones(string jewels, string stones) {
        set <char>jewls ( begin( jewels ) , end( jewels ) );
        unordered_map<char, int> umap;
        
        for(char s : stones) {
            umap[s]++;
        }
        
        int resp = 0;
        for(char j : jewls) {
            if(umap.find(j) != umap.end()) {
                resp += umap.at(j);
            }
        }
        return(resp);
    }
};
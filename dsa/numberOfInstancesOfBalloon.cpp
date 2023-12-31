#include <vector>;
#include <iostream>;
using namespace std;

class Solution {
public:
    int maxNumberOfBalloons(string text) {
        unordered_map<int, int> umap;
        for(int chr : text) {
            umap[chr]++;
        }
        
        int resp = 0;
        while(
            umap['b'] > 0 &&
            umap['a'] > 0 &&
            umap['l'] > 1 &&
            umap['o'] > 1 &&
            umap['n'] > 0
        ) {
            umap['b']-=1;
            umap['a']-=1;
            umap['l']-=2;
            umap['o']-=2;
            umap['n']-=1;
            resp++;
        }
        return(resp);
    }
};
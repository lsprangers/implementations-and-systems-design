#include <vector>;
using namespace std;

//997
class Solution {
public:
    int findJudge(int n, vector<vector<int>>& trust) {
        if(trust.size() < 1) {
            if(n == 1) {
                return(1);
            }
            return(-1);
        }


        unordered_map<int, int> outdegree;
        unordered_map<int, int> indegree;

        for(vector<int> trust_rel : trust) {
            indegree[trust_rel[1]]++;
            outdegree[trust_rel[0]]++;
        }

        for(auto [key, val] : indegree) {
            if(val == n - 1) {
                if(outdegree[key] == 0) {
                    return(key);
                }
                break;
            }
        }
        return(-1);

    }
};
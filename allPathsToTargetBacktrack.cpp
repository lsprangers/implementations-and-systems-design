#include <vector>;
//787
class Solution {
public:
    int to_find = -1;
    vector<vector<int>> allPathsSourceTarget(vector<vector<int>>& graph) {
        vector<vector<int>> resp;
        to_find = graph.size() - 1;
        vector<int> curr;
        curr.push_back(0);
        backtrack(curr, graph, resp);
        return(resp);
    }
    void backtrack(vector<int>& curr, vector<vector<int>>& graph, vector<vector<int>>& resp) {
        if (find(curr.begin(), curr.end(), to_find) != curr.end()) {
            resp.push_back(curr);
            return;
        }

        for (int neighbor : graph[curr[curr.size() - 1]]) {
            curr.push_back(neighbor);
            backtrack(curr, graph, resp);
            curr.pop_back();
        }
    }    
};
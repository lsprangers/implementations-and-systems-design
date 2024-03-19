// 1557
class Solution {
public:

    vector<int> findSmallestSetOfVertices(int n, vector<vector<int>>& edges) {
        vector<int> resp;
        unordered_map<int, int> in_degree;  
        for(vector<int> edge : edges) {
            in_degree[edge[1]]++;
        }
        for(int key = 0; key < n; key++) {
            if(in_degree[key] == 0) {
                resp.push_back(key);
            }
        }
        return(resp);
    }
};
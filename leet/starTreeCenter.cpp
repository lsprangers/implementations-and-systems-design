class Solution {
public:
    int findCenter(vector<vector<int>>& edges) {
        // Want to find node with > 1 edges
        // Greedy
        unordered_map<int, int> node_freq;

        for(vector<int> edge : edges) {
            node_freq[edge[0]]++;
            if(node_freq[edge[0]] > 1) {
                return(edge[0]);
            }
            node_freq[edge[1]]++;
            if(node_freq[edge[1]] > 1) {
                return(edge[1]);
            }
        }
        return(-1);
    }
};
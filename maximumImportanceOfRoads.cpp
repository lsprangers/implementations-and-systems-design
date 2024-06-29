// 2285
#include <ranges>

class Solution {
public:
    long long maximumImportance(int n, vector<vector<int>>& roads) {
        // greedy - nodes with most connections get highest number
        // O(n)
        unordered_map<int, int> freq;
        for(vector<int> road : roads) {
            freq[road[0]]++;
            freq[road[1]]++;
        }

        // order each of the nodes by their frequency
        // O(n)? O(n logn)...not O(n)
        map<int, vector<int>> freqs;
        for(auto [key, value] : freq) {
            freqs[value].push_back(key);
        }

        // Assign weight to a node
        // O(n)
        unordered_map<int, int> node_values;
        int weight = n;
        for(auto const& [key, value] : freqs | std::views::reverse) {
            for(int node : value) {
                node_values[node] = weight;
                weight--;
            }
        }

        // Calculate resp
        // O(n)
        long long resp = 0;
        for(vector<int> road : roads) {
            resp += node_values[road[0]] + node_values[road[1]];
        }


        // O(3n logn) time O(4n) space
        return(resp);
    }
};
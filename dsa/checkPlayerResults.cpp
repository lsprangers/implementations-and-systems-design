#include <vector>;
using namespace std;


class Solution {
public:
    vector<vector<int>> findWinners(vector<vector<int>>& matches) {
        unordered_map<int, int> losses;
        vector<int> all_wins;
        vector<int> one_loss;
        
        for(vector<int> game : matches) {
            if(losses.find(game[0]) == losses.end()) {
                losses[game[0]] = 0;
            }
            losses[game[1]]++;
        }
        for (auto [key, val]: losses) {
            if(val == 1) {
                one_loss.push_back(key);
            }
            else if(val == 0) {
                all_wins.push_back(key);
            }
        }
        
        sort(all_wins.begin(), all_wins.end());
        sort(one_loss.begin(), one_loss.end());
        
        vector<vector<int>> results = {all_wins, one_loss};
        return(results);
    }
};
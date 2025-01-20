#include <vector>;
using namespace std;

class Solution {
    public:
        vector<vector<int>> allPathsSourceTarget(vector<vector<int>>& graph) {

            int nodes = graph.size();
            vector<int> current;
            current.push_back(0);
            vector<vector<int>> paths;
            depthh(0, current, paths, nodes, graph);
            return paths;
        }
        void depthh(int i, vector<int> &current, vector<vector<int>> &paths, int &nodes, vector<vector<int>> &graph) {
            if(i == nodes - 1){
                paths.push_back(current);
            }
            else {
                for(vector<int>::iterator itv = graph[i].begin(); itv != graph[i].end(); ++itv){
                    vector<int> tmpcur = current;
                    tmpcur.push_back(*itv);
                    depthh(*itv, tmpcur, paths, nodes, graph);
                }
            }
        }
};
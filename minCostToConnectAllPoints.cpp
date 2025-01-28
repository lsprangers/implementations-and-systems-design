// 1584
class Solution {
public:

    struct CompareByIndex {
        int index;

        CompareByIndex(int idx) : index(idx) {}

        bool operator()(const vector<int>& a, const vector<int>& b) const {
            return a[index] > b[index]; // Change to '<' for ascending order
        }
    };

    unordered_map<int, vector<int>> neighbors;
    int minCostConnectPoints(vector<vector<int>>& points) {
        priority_queue<vector<int>, vector<vector<int>>, CompareByIndex> minHeap(CompareByIndex(2));
        for(int x = 0; x < points.size() - 1; x++) {
            for(int y = x + 1; y < points.size(); y++) {
                int dist = manhattan_distance(points[x], points[y]);
                vector<int> distStruct = {x, y, dist};
                minHeap.push(distStruct);
            }
        }
        
        int taken = 0;
        int resp = 0;
        
        while(taken < points.size() - 1) {
            vector<int> coordStruct = minHeap.top();
            minHeap.pop();
            if(!createsCycle(coordStruct[0], coordStruct[1])) {
                neighbors[coordStruct[1]].push_back(coordStruct[0]);
                neighbors[coordStruct[0]].push_back(coordStruct[1]);
                taken++;
                resp += coordStruct[2];
            }
        }
        return(resp);
    }
    int manhattan_distance(vector<int> x, vector<int> y) {
        return(
            abs(x[0] - y[0]) + abs(x[1] - y[1])
        );
    }

    bool createsCycle(int x, int y) {
        if(neighbors.empty()) {
            return(false);
        }
        set<int> seen;
        stack<int> stack;
        stack.push(y);
        while(!stack.empty()) {
            int node = stack.top();
            stack.pop();
            if(node == x) {
                return(true);
            }
            else {
                for(int neighbor : neighbors[node]) {
                    if(seen.find(neighbor) == seen.end()) {
                        stack.push(neighbor);
                        seen.insert(node);
                    }
                }
            }
        }

        return(false);

    }
};
// 210

class Solution {
public:
    vector<int> findOrder(int numCourses, vector<vector<int>>& prerequisites) {
        unordered_map<int, vector<int>> dependencies;
        vector<int> indegree = vector(numCourses, 0);
        
        // O(E)
        for(vector<int> prereq : prerequisites) {
            // Need a list of classes where we need to take 1 before we take 0
            dependencies[prereq[1]].push_back(prereq[0]);
            indegree[prereq[0]]++;
        }
        
        vector<int> resp;
        queue<int> q;
        
        // O(V)
        for(int course = 0; course < numCourses; course++) {
            if(indegree[course] == 0) {
                q.push(course);
            }
        }
        
        // O(V + E)
        while(!q.empty()) {
            int course = q.front();
            q.pop();
            resp.push_back(course);
            
            for(int nextCourse : dependencies[course]) {
                indegree[nextCourse]--;
                if(indegree[nextCourse] == 0) {
                    q.push(nextCourse);
                }
            }
        }
        
        if(resp.size() != numCourses) {
            resp = {};
        }
        return(resp);
        
    }
};
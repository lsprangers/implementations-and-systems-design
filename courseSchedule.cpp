// 207

class Solution {
public:
    bool canFinish(int numCourses, vector<vector<int>>& prerequisites) {
        // Basically need to check if there's a cycle
        // Topo sort won't work since in-degree will fail
        // - prerequisites means to take prerequisites[0] you must take prerequisites[1]
        unordered_map<int, vector<int>> dependencies;
        vector<int> indegree = vector(numCourses, 0);

        // O(E)
        for(vector<int> prereq : prerequisites) {
            // The course prerequisites[1] is a prereq for everything
            //  in it's array
            dependencies[prereq[1]].push_back(prereq[0]);
            indegree[prereq[0]]++;
        }

        // O(V)
        queue<int> zero_indegree;
        for(int course = 0; course < numCourses; course++) {
            if(indegree[course] < 1) {
                zero_indegree.push(course);
            }
        }

        int courses_seen = 0;
        while(!zero_indegree.empty()) {
            int course = zero_indegree.front();
            zero_indegree.pop();
            courses_seen++;

            for(int dependent_course : dependencies[course]) {

                indegree[dependent_course]--;
                if(indegree[dependent_course] < 1) {
                    zero_indegree.push(dependent_course);
                }
            }
        }

        return(courses_seen == numCourses);


    }
};
// 1700
class Solution {
public:
    int countStudents(vector<int>& students, vector<int>& sandwiches) {
        deque<int> students_deq(students.begin(), students.end());

        int follower = students_deq.size();
        int currSandwich = 0;
        
        while(!students_deq.empty() && currSandwich < sandwiches.size() && follower > 0) {
            int topPreference = students_deq.front();
            students_deq.pop_front();
            if(topPreference != sandwiches[currSandwich]) {
                follower--;
                students_deq.push_back(topPreference);
            }
            else {
                currSandwich++;
                follower = students_deq.size();
            }
        }
        return(students_deq.size());
    }
};
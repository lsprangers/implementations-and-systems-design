//1171

struct ListNode {
    int val;
    ListNode *next;
    ListNode() : val(0), next(nullptr) {}
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode *next) : val(x), next(next) {}
};

class Solution {
public:
    ListNode* removeZeroSumSublists(ListNode* head) {
        ListNode* front = new ListNode(0, head);
        ListNode* iter = front;
        unordered_map<int, ListNode*> prefixSum;
        int pfxSum = 0;
        while(iter != nullptr) {
            pfxSum += iter->val;
            prefixSum[pfxSum] = iter; //Duplicates will be replaced by last entry of certain pfxSum
            iter = iter->next;
        }

        pfxSum = 0;
        iter = front;

        while(iter != nullptr) {
            pfxSum += iter->val;
            iter->next = prefixSum[pfxSum]->next; // This part is what got me & I couldn't do
            iter = iter->next;
        }
        
        return(front->next);
    }
};
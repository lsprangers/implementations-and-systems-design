// stack only - 2816

/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
    ListNode* doubleIt(ListNode* head) {
        if(head == nullptr || head->val == 0) {
            return(head);
        }
        stack<ListNode*> stack;
        ListNode* tmp = head;
        while(tmp != nullptr) {
            stack.push(tmp);
            tmp = tmp->next;
        }
        
        int tens = 0;
        int remainder = 0;

        while(stack.size() > 0) {
            ListNode* curr = stack.top();
            int currVal = curr->val;
            stack.pop();
            currVal *= 2;
            curr->val = (currVal % 10) + remainder;
            remainder = floor(currVal / 10);
        }
        if(remainder > 0) {
            ListNode* newHead = new ListNode(remainder, head);
            return(newHead);
        }
        return(head);

    }
};
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
    ListNode* removeNodes(ListNode* head) {
        deque<ListNode*> deque;
        ListNode* tmp = head;

        while(tmp != nullptr) {
            while(deque.size() > 0 && tmp->val > deque.front()->val) {
                deque.pop_front();
            }
            deque.push_front(tmp);
            tmp = tmp->next;
        }

        ListNode* newHead = deque.back();
        deque.pop_back();
        ListNode* curr = newHead;
        while(deque.size() > 0) {
            ListNode* next = deque.back();
            deque.pop_back();
            curr->next = next;
            curr = next;         
        }
        curr->next = nullptr;
        
        return(newHead);
    }
};
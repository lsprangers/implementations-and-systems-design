#include <vector>;
#include <cmath>;
using namespace std;

//1290

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
    int getDecimalValue(ListNode* head) {
        vector<int> storage;
        while(head != nullptr) {
            storage.push_back(head->val);
            head = head->next;
        }
        int base = pow(2, storage.size() - 1);
        int resp = 0;
        for(int num : storage) {
            resp += num * base;
            base /= 2;
        }
        return(resp);
        
    }
};
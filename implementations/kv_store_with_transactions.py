
# - This is a quick demonstration of doing a KV Store with transactions.
# - The goal is to have a KV Store that supports transactions, and that
#    transactions are atomic and can be rolled back.
# - Our requirements are listed below, and our desires are that 
#    set, get, start_transaction, commit_transaction, rollback_transaction are all O(1) operations




from collections import deque
from typing import Union
from dataclasses import dataclass

# Requirements:
# 1. Set and get key-value pairs
# 2. Support transactions
#   2.1 - Start a transaction
#   2.2 - Commit a transaction
#       2.2.1 - Optional to pass a transaction ID and commit that specific transaction
#   2.3 - Rollback a transaction
#   2.4 - During a transaction, any set should be done in the context of the transaction, 
#           but get should not be affected by the transaction
#          i.e. set(a, 1) -> get(a) should return 1
#               start_transaction -> set(a, 2) -> get(a) should return 1
#               commit_transaction -> get(a) should return 2
# 3. Transactions should be atomic
# 4. Transactions should be able to be rolled back


@dataclass
class Transaction:
    transaction_id: int
    transaction_data: dict
    
class KVStoreWithTransactions:
    def __init__(self):
        self.kv_store = {}
        
        # deque:
        #   pros: O(1) append, O(1) pop
        #   cons: O(n) random access & delete from middle
        # list:
        #   pros: O(1) random access, O(1) append (unless resize), O(1) pop
        #   cons: O(n) delete from middle, resizing
        # self.transactions = deque()
        # map:
        #   pros: O(1) random access, delete, create
        #   cons: O(n) iteration...but we're going to keep a transaction ID and a latest transaction
        #       so we can just iterate through and transactions we need and get the location in O(1)
        #   For example, if we need to support committing latest transaction we can do that, or we can allow
        #       a user to specify a specific transaction ID - both are O(1)....we can also allow something
        #       like "specific iteration through latest iteration" by looping from specific -> latest and
        #       getting the transaction we need in O(1) 
        self.transactions = {}
        self.next_transaction_id = 0
    
    def start_transaction(self) -> int:
        # O(1) start transaction
        new_transaction = Transaction(self.next_transaction_id, {})
        self.next_transaction_id += 1
        self.transactions[new_transaction.transaction_id] = new_transaction
        return new_transaction.transaction_id

    def commit_transaction(self, transaction_id: int = - 1) -> bool:
        # O(1) commit transaction
        if len(self.transactions) < 1:
            return False
        
        # If they don't give us one, get latest
        if transaction_id == -1:
            tid_to_commit = self.get_latest_transaction_id()
        else:
            tid_to_commit = transaction_id
        
        try:
            transaction_to_commit = self.transactions[tid_to_commit]
        except KeyError:
            print(f"Transaction ID {tid_to_commit} not found")
            return False
        
        self.kv_store.update(transaction_to_commit.transaction_data)
        self.delete(transaction_to_commit.transaction_id)
        
        return True
    
    def rollback_transaction(self, transaction_id: int = -1) -> bool:
        if transaction_id == -1:
            tid_to_rollback = self.get_latest_transaction_id()
        else:
            tid_to_rollback = transaction_id
        
        try:
            self.transactions.pop(tid_to_rollback)
            return(True)
        except KeyError:
            print(f"Transaction ID {tid_to_rollback} not found")
            return False
        
    def get_latest_transaction_id(self) -> Union[None | int]:
        # O(1) get latest transaction
        if len(self.transactions) < 1:
            return None
        else:
            return self.next_transaction_id - 1
        
    def set(self, key, value) -> bool:
        # O(1) set
        
        latest_transaction_id = self.get_latest_transaction_id()
        if latest_transaction_id is not None:
            latest_transaction = self.transactions[latest_transaction_id]
            latest_transaction.transaction_data[key] = value
        else:
            self.kv_store[key] = value
        
        return(True)
    
    def get(self, key) -> Union[None | str]:
        # O(1) get
        if key not in self.kv_store:
            return(None)
        
        # If delete
        if self.kv_store[key] == -1:
            return None
        
        return(self.kv_store[key])

    def delete(self, key):
        '''For deletes we're just going to use tombstone -1 in case of rollbacks'''
        self.set(key, -1)

if __name__ == "__main__":
    test_store = KVStoreWithTransactions()
    assert test_store.get("a") == None
    test_store.set("a", 1)
    assert(test_store.get("a") == 1)
    test_store.delete("a")
    assert(test_store.get("a") == None)
    test_store.set("b", 2)

    t_id = test_store.start_transaction()
    assert(t_id == 0)
    test_store.set("a", 3)
    test_store.set("b", 3)
    assert(test_store.get("a") == None)
    assert(test_store.get("b") == 2)
    committed_first = test_store.commit_transaction()
    assert(committed_first == True)
    assert(test_store.get("a") == 3)
    assert(test_store.get("b") == 3)
    
    #  start a second transaction
    second_t_id = test_store.start_transaction()
    assert(second_t_id == 1)
    test_store.set("a", 4)
    test_store.set("b", 4)
    
    # start a third transaction
    third_t_id = test_store.start_transaction()
    assert(third_t_id == 2)
    test_store.set("a", 5)
    test_store.set("b", 5)
    
    # commit the third transaction before second
    committed_third = test_store.commit_transaction(third_t_id)
    assert(committed_third == True)
    assert(test_store.get("a") == 5)
    assert(test_store.get("b") == 5)
    
    # commit the second transaction
    committed_second = test_store.commit_transaction(second_t_id)
    assert(committed_second == True)
    assert(test_store.get("a") == 4)
    assert(test_store.get("b") == 4)
    
    # start a fourth transaction
    fourth_t_id = test_store.start_transaction()
    assert(fourth_t_id == 3)
    test_store.set("a", 6)
    test_store.set("b", 6)
    rolled_back = test_store.rollback_transaction(fourth_t_id)
    assert(rolled_back == True)
    assert(test_store.get("a") == 4)
    assert(test_store.get("b") == 4)
    
    # bad id's
    bad_id = 100
    assert(test_store.commit_transaction(bad_id) == False)
    assert(test_store.rollback_transaction(bad_id) == False)
    
    # random
    assert(test_store.get_latest_transaction_id() == 3)
    assert(test_store.get_latest_transaction_id() == 3)
    
    print("All tests pass :-)")
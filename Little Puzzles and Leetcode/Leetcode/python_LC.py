##### Two Sum #####
# O(n) time O(n) space
def twoSum(self, nums: List[int], target: int) -> List[int]:
        tab = {}
        for i,n in enumerate(nums):
            if target-n in tab:
                return [i,tab[target-n]]
            tab[n] = i
                

##### Add Two Numbers #####
# O(n) time O(1) space
def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        final = ListNode()
        curr = final
        carry = 0
        
        while True:
            if l1 is None and l2 is None:
                if carry == 1:
                    curr.val = 1
                else:
                    prev.next = None
                return final
            
            elif l1 is None:
                l1 = ListNode()
            elif l2 is None:
                l2 = ListNode()
            
            val = l1.val + l2.val + carry
            curr.val = val % 10
            carry = val // 10
            curr.next = ListNode()
            prev = curr
            curr = curr.next
            l1 = l1.next
            l2 = l2.next
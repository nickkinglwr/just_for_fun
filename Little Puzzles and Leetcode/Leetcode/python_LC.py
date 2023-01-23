##### Two Sum #####
# O(n) time O(n) space
def twoSum(self, nums: List[int], target: int) -> List[int]:
        tab = {}
        for i,n in enumerate(nums):
            if target-n in tab:
                return [i,tab[target-n]]
            tab[n] = i
                


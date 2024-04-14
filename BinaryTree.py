class BinaryTree:

    def __init__(self) -> None:
        pass

    class Node:

        def __init__(self,item) -> None:
            self.item = item
            self.left = None
            self.right = None

        def get_key(self):
            return self.item
        
        def is_leaf(self):
            return self.left == None and self.right == None
        
        def __str__(self) -> str:
            return str(self.item)
        

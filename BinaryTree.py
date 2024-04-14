class BinaryTree:
    """Binary search tree

    """

    def __init__(self,item=None) -> None:
        if item != None:  
            self.root = self.Node(item)
        else:
            self.root = None

    def add_node(self,item,current_node=None):
        """add node to tree
        Adds a node to the tree by searching for a free space recursively
        Args:
            item (int): value being stored
            current_node (Node, optional): the current node being searched. Defaults to None.
        """
        if self.root is None: #if there is no root, add the item as the root.
            self.root = self.Node(item)

        elif current_node == None: #If there is a root, begin the recursive search process.
            self.add_node(item,self.root)

        elif item >= current_node.get_item(): # If the item is greater than the current nodes item manipulate the right branches of that node.
            if current_node.get_right() is None:
                current_node.set_right(self.Node(item)) #set the right value to the new node if there is no right branch.
            else:
                self.add_node(item,current_node.get_right()) #Move to the right node if there is a right branch
        else:
            if current_node.get_left() is None:
                current_node.set_left(self.Node(item)) #set the left value to the new node if there is no left branch.
            else:
                self.add_node(item,current_node.get_left()) #Move to the left node if there is a left branch

    def search(self, item, current_node=None):
        """determine if an item is in the tree

        Args:
            item (int): value being searched for
            current_node (Node, optional): Current Node being examined. Defaults to None.

        Returns:
            Boolean: True if value si in the tree, false if not
        """
        if current_node is None:
            current_node = self.root

        current_item = current_node.get_item()

        if current_item == item:
            return True
        
        elif item > current_item and current_node.get_right():
            return self.search(item, current_node.get_right())
        
        elif item < current_item and current_node.get_left():
            return self.search(item, current_node.get_left())
        
        else:
            return False

    def find_item_depth(self, item, current_node=None):
        """find the depth of a value in the tree
        Uses recursion to determine how deep a value is in the tree.
        Args:
            item (int): value being searched for
            current_node (Node, optional): Current node being examined. Defaults to None.

        Returns:
            int: Depth of the value in the tree.
            str: if item isn't found return item not in tree
        """
        if not self.search(item):
            return "item not found"
        if current_node == None:
            current_node = self.root
        if self.root is None: #tree is empty
            return 0
        else:
            if current_node is not None:
                current_item = current_node.get_item() # get the value of the current node if it exists
            else:
                return 0
            if current_item == item: # if the item is found return a 1 otherwise return 1 plus the result of the next node.
                return 1
            elif current_item > item:
                return 1 + self.find_item_depth(item, current_node.get_left())
            elif current_item < item:
                return 1 + self.find_item_depth(item, current_node.get_right())
            
    def get_height(self,current_node=1):
        """get height of the tree
        recursively search for the highest branch in the tree.
        Args:
            current_node (Node, optional): Current node being examined. Defaults to None.

        Returns:
            int: height of highest branch
        """
        if current_node == 1:
            current_node = self.root

        if current_node is None:
            return 0

        if current_node.get_left() is None and current_node.get_right() is None:
            return 1

        else:
            right_count = self.get_height(current_node.get_right())
            left_count = self.get_height(current_node.get_left())

            if right_count > left_count:
                return 1 + right_count

            else:
                return 1 + left_count
            
    def count_nodes(self,current_node=1):
        """count all nodes in the tree
        Recursively search the tree to find the count of all nodes
        Args:
            current_node (Node, optional): current node being counted. Defaults to None.

        Returns:
            int: count of all nodes.
        """
        if current_node == 1:
            current_node = self.root

        if current_node is None:
            return 0
        
        elif current_node.get_left() is None and current_node.get_right() is None:
            return 1
        else:
            right_count = self.count_nodes(current_node.get_right())
            left_count = self.count_nodes(current_node.get_left())
            return 1 + right_count + left_count
        

    class Node:
        """Tree node
        Attributes
        ----------
        item: int
            value being stored
        left: Node
            Node with lower value
        right: Node
            Node with higher value        
        Methods
        -------
        is_leaf(None): Boolean
            returns wether node is a leaf node/ has no continuing branches.
        getters and setters
        __str__: string
            return stored item as str.
        """

        def __init__(self,item) -> None:
            """ Initialize object
            Create a Node object. set the item to the item passed and initiate no left and right nodes.

            Parameters
            ----------
            item: int
                item stored
            """
            self.item = item
            self.left = None
            self.right = None

        def is_leaf(self):
            """leaf node check
            Determine wether the node has any branches.

            return true if leaf false if not
            """
            return self.left == None and self.right == None
        
        def get_item(self):
            """get stored item

            Returns:
                int: value being stored in item
            """
            return self.item
        
        def get_left(self):
            """get left node

            Returns:
                Node: Lower valued node
            """
            return self.left
        
        def get_right(self):
            """get right node

            Returns:
                Node: Higher valued node
            """
            return self.right
        
        def set_left(self,left_node):
            """set left node

            Args:
                left_node (Node): Lower valued node
            """
            self.left = left_node

        def set_right(self,right_node):
            """set right node

            Args:
                right_node (Node): Higher valued node
            """
            self.right = right_node
        
        def __str__(self) -> str:
            """string representation

            Returns:
                str: item value as a string
            """
            return str(self.item)
        

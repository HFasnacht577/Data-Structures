class Node():

    def __init__(self,item):
        """
        The initializing method that defines the values of next,previous, and the node itself
        :param item: the item being assigned to the node
        """
        self.data = item
        self.next = None
        self.previous = None

    def get_data(self):
        """
        a method that returns the data held by the node
        :return: the nodes data
        """
        return self.data

    def get_next(self):
        """
        method to get the item assigned to the next value
        :return: the item in self.next
        """
        return self.next

    def get_previous(self):
        """
        method to get the item assigned to the previous value
        :return:  the item in self.previous
        """
        return self.previous

    def set_next(self,item):
        """
        method to change the self.next value
        :param item: the item the value is being changed to
        :return: nothing
        """
        self.next = item

    def set_previous(self,item):
        """
        method to change the self.previous value
        :param item: the item the value is being changed to
        :return: nothing
        """
        self.previous = item
        
    def __str__(self):
        """
        string method
        :return: the string of the nodes data
        """
        return str(self.data)
    
    def __repr__(self):
        return str(self.__str__())

class DLL():

    def __init__(self):
        """
        initializing method that defines a head and tail for the list
        """
        self.head = None
        self.tail = None

    def is_empty(self):
        """
        method to check whether a list is empty
        :return: True if empty, False otherwise
        """
        return self.head == None

    def add_to_head(self,item):
        """
        adds item to head of list
        :param item: item being added
        :return: nothing
        """

        temp = Node(item)
        temp.set_next(self.head)
        if self.length() > 0:
            self.head.set_previous(temp)
        self.head = temp
        if self.length() == 1:
            self.tail = self.head

    def pop_head(self):
        """
        removes and returns item at head of list
        :return: the value at the head of the list
        """
        value = self.head.get_data()
        new_head = self.head.get_next()
        new_head.set_previous(None)
        self.head = new_head
        return value

    def add_to_tail(self,item):
        """
        adds item to tail of list
        :param item: item being added
        :return: nothing
        """
        temp = Node(item)
        self.tail.set_next(temp)
        temp.set_previous(self.tail)
        self.tail = temp

    def pop_tail(self):
        """
        removes and returns item at tail of list
        :return: value at tail of list
        """
        value = self.tail.get_data()
        new_tail = self.tail.get_previous()
        new_tail.set_next(None)
        self.tail = new_tail
        return value

    def insert(self,index,item):
        """
        puts item in designated index in list
        :param index: the location for item
        :param item: the item being added
        :return: nothing
        """
        temp = Node(item)
        current = self.head
        count = 0
        while count != index:
            current = current.get_next()
            count += 1
        prev = current.get_previous()
        temp.set_next(current)
        temp.set_previous(prev)
        if prev != None:
            prev.set_next(temp)
        if prev == None:
            self.head = temp
        current.set_previous(temp)

    def remove(self,index):
        """
        removes item at specified index
        :param index: the location of item being removed
        :return: the item at index
        """
        current = self.head
        count = 0
        while count != index:
            current = current.get_next()
            count += 1
        next = current.get_next()
        prev = current.get_previous()
        next.set_previous(prev)
        prev.set_next(next)
        return current.get_data()

    def length(self):
        """
        finds the length of the list
        :return: the length of the list
        """
        current = self.head
        count = 0
        while current != None:
            count += 1
            current = current.get_next()
        return count

    def __str__(self):
        """
        checks each item in list and puts the values together to form a list
        :return: a list of items
        """
        final = "["
        current = self.head
        while current != None:
            if current.get_next() != None:
                final += str(current.get_data()) + "; "
                current = current.get_next()
            else:
                final += str(current.get_data())
                current = current.get_next()
        final += "]"
        return str(final)
    
    def __repr__(self):
        return str(self.__str__())

    def contains(self,item):
        """
        checks if the list contains the item
        :param item: the item being checked for
        :return: True if item is in list, False otherwise
        """
        current = self.head
        while current != None:
            if current.get_data() == item:
                return True
            else:
                current = current.get_next()
        return False

    def last_index(self,item):
        """
        finds the last index that the item is located at
        :param item: the item being checked for
        :return: the last location of the item
        """
        count = self.length() - 1
        current = self.tail
        while count != -1:
            if current.get_data() == item:
                return count
            else:
                current = current.get_previous()
                count -= 1
        return count
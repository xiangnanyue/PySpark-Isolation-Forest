# -*- coding: utf-8 -*-


# now write the function for create itree
class Node(object):
    def __init__(self, lefttree, righttree, splitattr, splitvalue, size=-1):
        self.left = lefttree
        self.right = righttree
        self.splitattr = splitattr
        self.splitvalue = splitvalue
        self.size = size
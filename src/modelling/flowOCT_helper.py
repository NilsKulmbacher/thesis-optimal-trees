"""
Defines `TreeStructure` class and the `make_tree` function to create tree and extract metrics.

Classes:
    TreeStructure: Represents a decision tree with methods to add, remove, and manage nodes.

        Nested Classes:
            Node: Class representing a node in the tree

Functions:
    make_tree: Creates a `TreeStructure` object from a Gurobi model and returns it

"""

from modelling.StrongTree.utils import get_node_status
import matplotlib.pyplot as plt
from graphviz import Digraph


class TreeStructure:

    class Node:
        def __init__(self, node_id, feature=None, value=None, is_leaf=False, depth=0):
            self.node_id = node_id
            self.depth = depth
            self.feature = feature
            self.value = value
            self.is_leaf = is_leaf
            self.children = []

        def __repr__(self):
            if self.is_leaf:
                return f"Leaf(Node {self.node_id}, Value: {self.value}, Children: {len(self.children)}, Depth: {self.depth})"
            else:
                return f"Branch(Node {self.node_id}, Feature: {self.feature}, Children: {len(self.children)}), Depth: {self.depth})"

    def __init__(self):
        self.nodes = {}
        self.root = None

    def add_node(self, node_id, feature=None, value=None, is_leaf=False, parent_id=0, depth=0):
        new_node = self.Node(node_id, feature, value, is_leaf, depth)
        self.nodes[node_id] = new_node
        if parent_id != 0:
            self.nodes[parent_id].children.append(new_node)
        else:
            self.root = new_node
    
    def remove_node(self, node_id):
        node = self.nodes[node_id]
        if node_id != self.root.node_id:
            for parent_node in self.nodes.values():
                if node in parent_node.children:
                    parent_node.children.remove(node)
        del self.nodes[node_id]

    # Print structured tree for debugging
    def print_tree(self, node=None, level=0):
        if node is None:
            node = self.root
        indent = "   " * level
        print(indent + repr(node))
        for child in node.children:
            self.print_tree(child, level + 1)

    def __str__(self):
        return f"Decision Tree with {len(self.nodes)} nodes"

    def num_nodes(self):
        return len(self.nodes)

    def num_leaves(self):
        return len([leaf for leaf in self.nodes if self.nodes[leaf].is_leaf])

    def max_depth(self):
        return max([self.nodes[leaf].depth for leaf in self.nodes if self.nodes[leaf].is_leaf])

    def avg_depth(self):
        return sum([self.nodes[leaf].depth for leaf in self.nodes if self.nodes[leaf].is_leaf])/self.num_leaves()

    def features_used(self):
        return len(set([self.nodes[branch].feature for branch in self.nodes if not self.nodes[branch].is_leaf]))

    def plot_tree(self):
        def add_edges(dot, node):
            for child in node.children:
                dot.edge(str(node.node_id), str(child.node_id),
                         label="True" if child == node.children[0] else "False")
                add_edges(dot, child)

        def add_nodes(dot, node):
            if node.is_leaf:
                dot.node(str(node.node_id), label=f'Predict: {node.value}', shape='ellipse')
            else:
                dot.node(str(node.node_id), label=f'{node.feature}=0', shape='oval')
            for child in node.children:
                add_nodes(dot, child)

        dot = Digraph()
        add_nodes(dot, self.root)
        add_edges(dot, self.root)
        return dot

    # Custom predict function
    def predict(self, X_test):
        predictions = []
        self.print_tree()
        for index, sample in X_test.iterrows():
            node = self.root
            while not node.is_leaf:
                if sample[node.feature] == 0:
                    node = node.children[0]
                else:
                    node = node.children[1]
            predictions.append(node.value)

        return predictions

# Create a tree structure from a Gurobi model
def make_tree(grb_model, b, beta, p):
    tree = TreeStructure()
    model_tree = grb_model.tree
    for n in model_tree.Nodes + model_tree.Leaves:
        pruned, branching, selected_feature, leaf, value = get_node_status(
            grb_model, b, beta, p, n)
        parent_id = int(model_tree.get_parent(n))
        depth = len(model_tree.get_ancestors(n))
        if pruned:
            continue
        elif branching:
            tree.add_node(node_id=n, feature=selected_feature,
                          is_leaf=False, parent_id=parent_id, depth=depth)
        elif leaf:
            tree.add_node(node_id=n, value=value, is_leaf=True,
                          parent_id=parent_id, depth=depth)

    tree.print_tree()
    return tree

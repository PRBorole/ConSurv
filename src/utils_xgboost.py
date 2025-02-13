import numpy as np
import pandas as pd

import xgboost as xgb
from xgboost import plot_tree


def parse_tree(tree_str):
    def parse_node(nodes_dict, node_id, conditions=[]):
        node = nodes_dict[node_id]

        if "leaf=" in node:
            value = node.split("leaf=")[-1].strip()
            conditions_str = " and ".join(conditions)
            rules.append(f"IF {conditions_str} THEN {value}")
            return

        condition = node.split(']')[0] + ']'
        yes_branch = node.split('yes=')[1].split(',')[0].strip()
        no_branch = node.split('no=')[1].split(',')[0].strip()
        missing_branch = node.split('missing=')[1].split('\n')[0].strip()

        yes_condition = conditions + [condition]
        no_condition = conditions + [condition.replace('<', '>=')]

        parse_node(nodes_dict, yes_branch, yes_condition)
        parse_node(nodes_dict, no_branch, no_condition)

    # Split the input string into nodes, preserving the tab characters
    nodes = tree_str.split('\n')
    nodes_dict = {}
    for node in nodes:
        if ':' in node:
            node_id, node_content = node.split(':', 1)
            nodes_dict[node_id.strip()] = node_content.strip()

    # List to store the rules
    rules = []

    # Start parsing from the root node (node 0)
    parse_node(nodes_dict, '0')

    return rules
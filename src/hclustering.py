from parser import parse_header, parse_data
from argparse import ArgumentParser
from os import path
from datetime import datetime
import numpy as np
import sys
import json
import pdb

class Node:
    def __init__(self, node_type, data=None):
        if node_type not in ['LEAF', 'NODE', 'ROOT']:
            raise Exception("invalid node type: {}".format(node_type))
        self.node_type = node_type
        self.data = data

    def to_dict(self, height):
        node = {
            "type": self.node_type,
            "height": height,
        }
        if self.node_type == "LEAF":
            node["data"] = list(self.data)
        else:
            node["nodes"] = [child.to_dict(height + 1) for child in self.data]
        return node


def single_link(c1, c2):
    min_dist = sys.float_info.max
    for a in c1:
        for b in c2:
            dist = sum(abs(a - b))
            if dist < min_dist:
                min_dist = dist
    return min_dist


def update_dist(data, clusters, dist_mat, dist_func):
    closest = ((-1, -1), sys.float_info.max)
    for ai in range(0, len(clusters)):
        for bi in range(ai + 1, len(clusters)):
            c1 = [data.values[i] for i in clusters[ai]]
            c2 = [data.values[i] for i in clusters[bi]]
            dist_mat[ai][bi] = dist_func(c1, c2)
            if dist_mat[ai][bi] < closest[1]:
                closest = ((ai, bi), dist_mat[ai][bi])
    return closest[0]


def generate(data, dist_func):
    clusters = [(di,) for di in range(0, len(data))]
    tree = [Node("LEAF", clusters[ci]) for ci in range(0, len(clusters))]
    dist_mat = np.zeros((len(clusters), len(clusters)), dtype=np.float64)
    while len(clusters) > 1:
        closest = update_dist(data, clusters, dist_mat, dist_func)
        consumer, consumed = (closest[0], closest[1]) if len(clusters[closest[0]]) >= len(clusters[closest[1]]) else (closest[1], closest[0])
        tree[consumer] = Node("NODE", (tree[consumer], tree[consumed]))
        tree.remove(tree[consumed])
        clusters[consumer] = clusters[consumer] + clusters[consumed]
        clusters.pop(consumed)
    tree[0].node_type = "ROOT"
    return tree[0]


if __name__ == "__main__":
    parser = ArgumentParser(description="A heirarchical clustering program using the agglomeration method.")
    parser.add_argument("filename", help="Name of input file in csv form. Note that the first line of this file must be a restrictions vector.")
    parser.add_argument("--threshold", help="Specify a threshold value for which to stop agglomeration. By default, will produce the full cluster heirachy")
    parser.add_argument("--link-method", default="SINGLE", help="Specify link-method for agglomeration. Allowed values: SINGLE | COMPLETE | AVERAGE | WARD]. By default SINGLE.")
    parser.add_argument("--headers", help="Name of input file containing header names on a single line in CSV format. Ommitting this will produce plots with unnamed axis")
    args = parser.parse_args()
    headers = None
    if args.headers:
        headers = parse_header(args.headers)
    data = parse_data(args.filename, headers)

    # Generate heirarchy tree
    root = generate(data, single_link)

    # Write tree to JSON file
    timestamp = "_" + str(datetime.now().replace(microsecond=0)).replace(' ', '_').replace(':', '-')
    tree_filename = path.basename(args.filename).split('.')[0] + timestamp + ".json"
    with open(tree_filename, 'w') as file:
        file.write(json.dumps(root.to_dict(0), indent=4, separators=(',', ': ')))



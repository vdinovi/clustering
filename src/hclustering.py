from parser import parse_header, parse_data
from argparse import ArgumentParser
from os import path
from datetime import datetime
import numpy as np
from pprint import pprint
import sys
import json
import pdb

class Node:
    def __init__(self, node_type, height, data=None):
        if node_type not in ['LEAF', 'NODE', 'ROOT']:
            raise Exception("invalid node type: {}".format(node_type))
        self.node_type = node_type
        self.data = data
        self.height = float(height)

    def to_dict(self):
        node = {
            "type": self.node_type,
            "height": self.height,
        }
        if self.node_type == "LEAF":
            node["data"] = self.data
        else:
            node["nodes"] = [child.to_dict() for child in self.data]
        return node

def single_link(c1, c2, dist_mat):
    for row in range(0, len(dist_mat)):
        dist_mat[row, c1] = min(dist_mat[row, c1], dist_mat[row, c2])
    dist_mat = np.delete(dist_mat, c2, axis=0)
    dist_mat = np.delete(dist_mat, c2, axis=1)
    return dist_mat

def complete_link(c1, c2, dist_mat):
    for row in range(0, len(dist_mat)):
        dist_mat[row, c1] = max(dist_mat[row, c1], dist_mat[row, c2])
    dist_mat = np.delete(dist_mat, c2, axis=0)
    dist_mat = np.delete(dist_mat, c2, axis=1)
    return dist_mat

#def average_link(c1, c2, dist_mat):
#    total = 0.0
#    count = 0
#    for a in c1:
#        total += sum([dist_mat[b] for b in c2])
#    return total / count
#def centroid(c1, c2):
#    centroid1 = sum(c1) / float(len(c1))
#    centroid2 = sum(c2) / float(len(c2))
#    return sum(abs(centroid2 - centroid1))
#def wards(c1, c2):
#    centroid1 = sum(c1) / float(len(c1))
#    centroid2 = sum(c2) / float(len(c2))
#    coef = ((len(c1) * len(c2)) / float(len(c1) + len(c2)))
#    return coef * sum((centroid2 - centroid1)) ** 2

def distance_matrix(data):
    clusters = [(di,) for di in range(0, len(data))]
    dist_mat = np.zeros((len(clusters), len(clusters)), dtype=np.float64)
    for ai in range(0, len(clusters)):
        for bi in range(ai, len(clusters)):
            dist_mat[ai, bi] = sum(abs((data.values[ai] - data.values[bi])))
    return dist_mat, clusters

def find_closest(dist_mat):
    x, y, dist = 0, 0, sys.float_info.max
    for ai in range(0, len(dist_mat)):
        for bi in range(ai, len(dist_mat)):
            if ai != bi and dist_mat[ai, bi] <= dist:
                dist = dist_mat[ai, bi]
                x, y = ai, bi
    return x, y, dist

def generate(data, dist_func):
    dist_mat, clusters = distance_matrix(data)
    tree = [Node("LEAF", 0, clusters[ci][0]) for ci in range(0, len(clusters))]
    while len(clusters) > 1:
        c1, c2, dist = find_closest(dist_mat)
        tree[c1] = Node("NODE", dist, (tree[c1], tree[c2]))
        tree.remove(tree[c2])
        clusters[c1] += clusters[c2]
        clusters.pop(c2)
        dist_mat = dist_func(c1, c2, dist_mat)
    tree[0].node_type = "ROOT"
    return tree[0]

def gather(node):
    if node.node_type == "LEAF":
        return [node.data]
    else:
        cluster = []
        for child in node.data:
            cluster += gather(child)
        return cluster


def get_clusters(node, threshold, clusters):
    if node.node_type == "LEAF":
        clusters.append([node.data])
    elif node.height <= threshold:
        clusters.append(gather(node))
    else:
        for child in node.data:
            get_clusters(child, threshold, clusters)


if __name__ == "__main__":
    parser = ArgumentParser(description="A heirarchical clustering program using the agglomeration method. Uses regular manhatten distance for point-distance calucations.")
    parser.add_argument("filename", help="Name of input file in csv form. Note that the first line of this file must be a restrictions vector.")
    parser.add_argument("--threshold", help="Specify a threshold value for which to stop agglomeration. By default, will produce the full cluster heirachy")
    parser.add_argument("--link-method", default="SINGLE", help="Specify link-method for agglomeration. Allowed values: SINGLE | COMPLETE | AVERAGE | CENTROID | WARD]. By default SINGLE.")
    parser.add_argument("--headers", help="Name of input file containing header names on a single line in CSV format. Ommitting this will produce plots with unnamed axis")
    args = parser.parse_args()
    headers = None
    if args.headers:
        headers = parse_header(args.headers)
    data = parse_data(args.filename, headers)

    link_methods = {
        "SINGLE": single_link,
        "COMPLETE": complete_link
        #"AVERAGE": average_link
        #"CENTROID": centroid,
        #"WARD": wards
    }

    # Generate heirarchy tree
    root = generate(data, link_methods[args.link_method])

    # Write tree to JSON file
    timestamp = "_" + str(datetime.now().replace(microsecond=0)).replace(' ', '_').replace(':', '-')
    tree_filename = path.basename(args.filename).split('.')[0] + timestamp + ".json"
    with open(tree_filename, 'w') as file:
        print("-> writing dendogram to {}".format(tree_filename))
        file.write(json.dumps(root.to_dict(), indent=4, separators=(',', ': ')))

    if args.threshold:
        clusters = []
        get_clusters(root, float(args.threshold), clusters)
        print("{} Clusters".format(len(clusters)))
        for c in clusters:
            print("   {}".format(c))






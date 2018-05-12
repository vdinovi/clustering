from parser import parse_header, parse_data
from stats import cluster_stats

from argparse import ArgumentParser
from os import path
from datetime import datetime
import numpy as np
import math
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

def single_link(c1, c2, dist_mat, clusters, data):
    # clusters, data unused -- passed for consistency with other methods
    for row in range(0, len(dist_mat)):
        dist_mat[row, c1] = min(dist_mat[row, c1], dist_mat[row, c2])
        dist_mat[c1, row] = dist_mat[row, c1]
    dist_mat = np.delete(dist_mat, c2, axis=0)
    dist_mat = np.delete(dist_mat, c2, axis=1)
    return dist_mat

def complete_link(c1, c2, dist_mat, clusters, data):
    # clusters, data unused -- passed for consistency with other methods
    for row in range(0, len(dist_mat)):
        dist_mat[row, c1] = max(dist_mat[row, c1], dist_mat[row, c2])
        dist_mat[c1, row] = dist_mat[row, c1]
    dist_mat = np.delete(dist_mat, c2, axis=0)
    dist_mat = np.delete(dist_mat, c2, axis=1)
    return dist_mat


def average_link(c1, c2, dist_mat, clusters, data):
    # clusters, data unused -- passed for consistency with other methods
    for row in range(0, len(dist_mat)):
        dist_mat[row, c1] = (dist_mat[row, c1] + dist_mat[row, c2]) / 2
        dist_mat[c1, row] = dist_mat[row, c1]
    dist_mat = np.delete(dist_mat, c2, axis=0)
    dist_mat = np.delete(dist_mat, c2, axis=1)
    return dist_mat

def centroid(c1, c2, dist_mat, clusters, data):
    # This methods require reading the data to compute centroids
    cluster = clusters[c1] + clusters[c2]
    centroid = sum([data.values[p] for p in cluster]) / len(cluster)
    for row in range(0, len(dist_mat)):
        orig = sum([data.values[p] for p in clusters[row]]) / len(clusters[row])
        dist_mat[row, c1] = sum(abs(orig - centroid))
        dist_mat[c1, row] = dist_mat[row, c1]
    dist_mat = np.delete(dist_mat, c2, axis=0)
    dist_mat = np.delete(dist_mat, c2, axis=1)
    return dist_mat

def wards(c1, c2, dist_mat, clusters, data):
    # This methods require reading the data to compute centroids
    cluster = clusters[c1] + clusters[c2]
    centroid = sum([data.values[p] for p in cluster]) / len(cluster)
    for row in range(0, len(dist_mat)):
        orig = sum([data.values[p] for p in clusters[row]]) / len(clusters[row])
        coef = len(clusters[row]) * len(cluster) / float(len(clusters[row]) + len(cluster))
        dist_mat[row, c1] = coef * math.sqrt(sum(orig - centroid) ** 2)
        dist_mat[c1, row] = dist_mat[row, c1]
    dist_mat = np.delete(dist_mat, c2, axis=0)
    dist_mat = np.delete(dist_mat, c2, axis=1)
    return dist_mat

def distance_matrix(data):
    clusters = [(di,) for di in range(0, len(data))]
    dist_mat = np.zeros((len(clusters), len(clusters)), dtype=np.float64)
    for ai in range(0, len(clusters)):
        for bi in range(0, len(clusters)):
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
        dist_mat = dist_func(c1, c2, dist_mat, clusters, data)
        tree[c1] = Node("NODE", dist, (tree[c1], tree[c2]))
        tree.remove(tree[c2])
        clusters[c1] += clusters[c2]
        clusters.pop(c2)
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

def plot_stats(filename, root, data, data_name, link_method):
    import matplotlib.pyplot as plt
    thresholds = []
    variances = []
    sizes = []

    thresh = 0
    dt = 0.1
    while thresh <= root.height * 1.25:
        clusters = []
        get_clusters(root, thresh, clusters)
        stats = cluster_stats(clusters, data)
        avg_var = sum([cluster["Dist-Variance"] for _, cluster in stats.items()]) / len(stats)
        num_clusters = len(stats)
        thresholds.append(thresh)
        variances.append(avg_var)
        sizes.append(num_clusters)
        thresh += dt
    plt.style.use('ggplot')
    plt.clf()
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    fig.suptitle('Heirarchical Clustering on {}'.format(data_name))
    fig.text(0.9, 0.9, "link method: " + link_method, fontsize=12)
    # Variance
    ax1.plot(thresholds, variances)
    ax1.set(ylabel="Avg. variance dist-to-center")
    # Size
    ax2.plot(thresholds, sizes)
    ax2.set(ylabel="Number of clusters", xlabel="threshold")
    print("-> writing stats plot to {}".format(filename))
    plt.savefig(filename)


def plot_clusters(filename, clusters, data, data_name):
    import matplotlib.pyplot as plt
    x, y, c = zip(*([(data.values[p][0], data.values[p][1], label) for label, cl in enumerate(clusters) for p in cl]))
    plt.style.use('ggplot')
    plt.clf()
    plt.title('Heirarchical Clustering on {}'.format(data_name))
    plt.scatter(x, y, c=c)
    print("-> writing cluster plot to {}".format(filename))
    plt.savefig(filename)





if __name__ == "__main__":
    parser = ArgumentParser(description="A heirarchical clustering program using the agglomeration method. Uses regular manhatten distance for point-distance calucations.")
    parser.add_argument("filename", help="Name of input file in csv form. Note that the first line of this file must be a restrictions vector.")
    parser.add_argument("--threshold", help="Specify a threshold value for which to stop agglomeration. By default, will produce the full cluster heirachy")
    parser.add_argument("--link-method", default="SINGLE", help="Specify link-method for agglomeration. Allowed values: SINGLE | COMPLETE | AVERAGE | CENTROID | WARDS]. By default SINGLE.")
    parser.add_argument("--headers", help="Name of input file containing header names on a single line in CSV format. Ommitting this will produce plots with unnamed axis")
    parser.add_argument("--plot-stats", action="store_true", help="Plot various statistics against varying threshold values")
    parser.add_argument("--plot-clusters", action="store_true", help="Plot clusters generated on 2-d data (note that this works only on 2-d data)")
    args = parser.parse_args()
    headers = None
    if args.headers:
        headers = parse_header(args.headers)
    data = parse_data(args.filename, headers)

    link_methods = {
        "SINGLE": single_link,
        "COMPLETE": complete_link,
        "AVERAGE": average_link,
        "CENTROID": centroid,
        "WARDS": wards
    }

    # Generate heirarchy tree
    root = generate(data, link_methods[args.link_method])

    # Write tree to JSON file
    timestamp = "_" + str(datetime.now().replace(microsecond=0)).replace(' ', '_').replace(':', '-')
    tree_filename = path.basename(args.filename).split('.')[0] + timestamp + ".json"
    with open(tree_filename, 'w') as file:
        print("-> writing dendrogram to {}".format(tree_filename))
        file.write(json.dumps(root.to_dict(), indent=4, separators=(',', ': ')))

    # Apply threshold to get resulting clusters
    if args.threshold:
        np.set_printoptions(precision=4)
        clusters = []
        get_clusters(root, float(args.threshold), clusters)
        # Print cluster stats
        stats = cluster_stats(clusters, data)
        for name, stat in stats.items():
            print(name)
            for k, v in stat.items():
                print("  {}: {}".format(k, v))
            print()
        # print points
        if args.plot_clusters:
            if len(data.columns) != 2:
                raise Exception("Cannot plot data with dimensions {}, must be 2".format(len(data.columns)))
            plot_filename = path.basename(args.filename).split('.')[0] + "_clusters" + timestamp + ".png"
            plot_clusters(plot_filename, clusters, data, args.filename)



    # Plot stats against thresholds
    if args.plot_stats:
        plot_filename = path.basename(args.filename).split('.')[0] + "_stats" + timestamp + ".png"
        plot_stats(plot_filename, root, data, args.filename, args.link_method)






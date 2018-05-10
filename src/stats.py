import pdb
import random

def centroid(cluster, data):
    return sum([data.values[p] for p in cluster]) / float(len(cluster))

def sample(cluster, data, limit):
    return random.sample([data.values[p] for p in cluster], limit)

def max_dist(cluster, data, center):
    return max([sum(abs(data.values[p] - center)) for p in cluster])

def min_dist(cluster, data, center):
    return min([sum(abs(data.values[p] - center)) for p in cluster])

def avg_dist(cluster, data, center):
    return sum([sum(abs(data.values[p] - center)) for p in cluster]) / float(len(cluster))

def cluster_stats(clusters, data):
    stats = {}
    for index, cluster in enumerate(clusters):
        center = centroid(cluster, data)
        stats["Cluster{}".format(index)] = {
            "Center": tuple(round(x, 4) for x in center),
            "Max Dist": round(max_dist(cluster, data, center), 4),
            "Min Dist": round(min_dist(cluster, data, center), 4),
            "Avg Dist": round(avg_dist(cluster, data, center), 4),
            "Size": len(cluster),
            "Sample": [tuple(s) for s in sample(cluster, data, 5)]
        }
    return stats


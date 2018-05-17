import numpy as np
import pandas as pd
from parser import *
from random import sample, randint
from math import sqrt
from stats import cluster_stats
from argparse import ArgumentParser
from datetime import datetime
from os import path
import pdb

def getTestData():
   return parse_data("../dataset/4clusters.csv")

def getInitialCentroids(data, k):
   #random centroids
   indices = sample(range(len(data)),k)
   d = {i: data.values[indices[i]] for i in range(k)}
   #d = {i:[ randint(mins[i],maxs[i]+1) for key in range(k)] for i in range(len(maxs))}
   df = pd.DataFrame(data=d).T
   return df

def distance(pointA,pointB,distanceType):
   sm = 0
   if(distanceType.lower() == 'manhattan'):
      for i in range(len(pointA)):
         sm += abs(pointA[i] - pointB[i])
      return np.float64(sm)
   else:
      for i in range(len(pointA)):
         sm+= pow(pointA[i]-pointB[i],2)
      return sqrt(sm)

def classify(data,centroids,distanceType):
   lst = []
   for pointA in data.values:
      bestCentroid = None
      bestDistance = None
      for i in range (len(centroids.values)):
         centroid = centroids.values[i]
         dist = distance(pointA,centroid,distanceType)
         if(bestDistance == None or dist < bestDistance):
            bestCentroid = i
            bestDistance = dist
      lst.append(bestCentroid)
   return lst

def centroidDifference(newCentroids,oldCentroids,distanceType):
   totalDistance = 0
   for i in range(len(newCentroids.values)):
      centroidA = newCentroids.values[i]
      centroidB = oldCentroids.values[i]
      totalDistance += distance(centroidA,centroidB,distanceType)
   return totalDistance

def updateCentroids(data,clusterValues,k):
   # should return a new set of centroids based on cluster values
   d = {key:[] for key in range(k)}
   for i in range(len(data.values)):
      d[clusterValues[i]].append(data.values[i])
   averaged = {key:np.average(d[key],axis=(0)) for key in d.keys()}
   newCentroids = [averaged[key] for key in averaged.keys()]
   return pd.DataFrame(newCentroids)

def kmeansHelper(data,k,distanceType,threshold):
   stopCondition = False
   # 1. get Initial centroids
   centroids = getInitialCentroids(data,k)
   lastCentroids = pd.DataFrame([])
   while(not stopCondition):
      # 2. Assign clusters
      clusterValues = classify(data,centroids,"manhattan")
      # 3. Update centroids
      newCentroids = updateCentroids(data,clusterValues,k)
      # Evaluate if we should stop
      #print(lastCentroids)
      if(centroidDifference(newCentroids,centroids,distanceType) < threshold):
         stopCondition = True
      elif ((not lastCentroids.empty) and centroidDifference(lastCentroids,newCentroids,distanceType)<threshold):
         stopCondition = True
      lastCentroids = centroids
      centroids = newCentroids
   return centroids,clusterValues

def kmeans(data,k,distanceType,threshold):
   centroids, c = kmeansHelper(data,k,distanceType,threshold);
   arr = [[i for i in range(len(c)) if c[i] == j] for j in set(c)]
   return arr

def plot_clusters(filename, clusters, data, data_name,k):
    import matplotlib.pyplot as plt
    x, y, c = zip(*([(data.values[p][0], data.values[p][1], label) for label, cl in enumerate(clusters) for p in cl]))
    plt.style.use('ggplot')
    plt.clf()
    plt.title('Kmeans Clustring on {} with k={}'.format(data_name,k))
    plt.scatter(x, y, c=c)
    print("-> writing cluster plot to {}".format(filename))
    plt.savefig(filename)

def plot_clusters_3d(filename, clusters, data, data_name,k):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    x, y, z, c = zip(*([(data.values[p][0], data.values[p][1], data.values[p][2], label) for label, cl in enumerate(clusters) for p in cl]))
    plt.style.use('ggplot')
    plt.clf()
    plt.title('Heirarchical Clustering on {} with k={}'.format(data_name,k))
    ax = Axes3D(plt.figure())
    ax.scatter(x, y, z, c=c)
    print("-> writing cluster plot to {}".format(filename))
    plt.savefig(filename)

def plot_stats(filename, data, data_name):
   import matplotlib.pyplot as plt
   threshold = .1
   ks = []
   variances = []
   min_var = float('inf')
   best_k = None
   for k in range(1,11): #
      clusters = kmeans(data,args.k,"manhattan",threshold)
      stats = cluster_stats(clusters,data)
      num_clusters = len(stats)
      avg_var = sum([cluster["Dist-Variance"] for _, cluster in stats.items()])/num_clusters
      
      if(avg_var < min_var):
         min_var = avg_var
         best_k = k

      ks.append(k)
      variances.append(avg_var)
   plt.style.use('ggplot')
   plt.clf()
   plt.title("Kmeans on {}".format(data_name))
   plt.plot(ks,variances)
   plt.xlabel("k")
   plt.ylabel("variance")
   plt.savefig(filename)

if __name__ == "__main__":
   parser = ArgumentParser(description="kmeans clustering.")
   parser.add_argument("filename", help="Name of the input file in csv form. Note that the first line of this file must be a restrictions vector.")
   parser.add_argument("--threshold",type=float, help="threshold for difference between clusters to end computation.")
   parser.add_argument("--k",type=int, help="number of clusters to find.")
   parser.add_argument("--headers", help="Name of input file containing header names on a single line in CSV format. Ommitting this will produce plots with unnamed axis")
   parser.add_argument("--plot-stats", action="store_true", help="Plot various statistics against varying threshold values")
   parser.add_argument("--plot-clusters", action="store_true", help="Plot clusters generated on 2-d data (note that this works only on 2-d data)")
   args = parser.parse_args()
  
   headers = None
   threshold = .1
   if args.threshold:
      threshold = args.threshold
   if args.headers:
      headers = parse_header(args.headers)
   data = parse_data(args.filename)
   clusters = kmeans(data,args.k,"manhattan",threshold)
   stats = cluster_stats(clusters,data)
   timestamp = "+" + str(datetime.now().replace(microsecond=0)).replace(' ', '_').replace(':', '-')
   for name, stat in stats.items():
      print(name)
      for k,v in stat.items():
         print("  {}: {}".format(k, v))
      print()
   if args.plot_stats:
      plot_filename = path.basename(args.filename).split('.')[0] + "_stats" + timestamp + ".png"
      plot_stats(plot_filename, data, args.filename)
   if args.plot_clusters:
      plot_filename = path.basename(args.filename).split(".")[0] + "_clusters"+timestamp+".png"
      if len(data.columns) == 3:
         plot_clusters_3d(plot_filename,clusters,data,args.filename,args.k)
      elif len(data.columns) == 2:
         plot_clusters(plot_filename,clusters,data,args.filename,args.k)
      else:
         raise Exception("Cannot plot data with {} dimension, must be 2".format(len(data.columns)))

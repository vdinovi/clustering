import numpy as np
import pandas as pd
from parser import *
from random import randint
from math import sqrt
from stats import cluster_stats
from argparse import ArgumentParser
from os import path
from datetime import datetime
import pdb

def getTestData():
   return parse_data("../dataset/4clusters.csv")

def distance(pointA,pointB,distanceType):
   sm = 0
   if(distanceType.lower() == 'manhattan'):
      for i in range(len(pointA)):
         sm += abs(pointA[i] - pointB[i])
      return float(sm)
   else:
      for i in range(len(pointA)):
         sm+= pow(pointA[i]-pointB[i],2)
      return sqrt(sm)

def nearbyPoints(data,i,e,distanceType):
   d = data.values
   return [idx for idx in range(len(d)) if idx != i and distance(d[idx],d[i],distanceType) < e]

def getCorePoints(data,numPoints,e,distanceType):
   corePoints = []
   for i in range(len(data.values)):
      n = nearbyPoints(data,i,e,distanceType)
      if(len(n) >= numPoints):
         corePoints.append(i)
   return corePoints

def addLabel(i,labels,cluster):
    if not labels[i] or np.isnan(labels[i]):
      labels[i] = cluster
      return True
    return False

def densityConnected(data,i,e,distanceType,cluster,core,labels):
   nb = nearbyPoints(data,i,e,distanceType)
   for i in nb: 
      if i in core and addLabel(i,labels,cluster):
         densityConnected(data,i,e,distanceType,cluster,core,labels)

def dbscanHelper(data,numPoints,e,distanceType):
   # Return clusters as an array of arrays, with each array containing a cluster
   core = getCorePoints(data,numPoints,e,distanceType)
   cluster = 1
   labels = np.full(len(data), None)
   for i in range(len(core)):
      if(addLabel(core[i],labels,cluster)):
         densityConnected(data,core[i],e,distanceType,cluster,core,labels)
         cluster+=1
   noise = [i for i in range(len(labels)) if not labels[i]]
   border = [i for i in range(len(labels)) if (i not in noise) and (i not in core)]
   clusters = [[i for i in range(len(labels)) if labels[i] == j] for j in set(labels)]
   clusters = [c for c in clusters if c]
   pdb.set_trace()
   return noise, border, core, clusters
   
def dbscan(data,numPoints,e,distanceType):
    noise, border, core, clusters = dbscanHelper(data,numPoints,e,distanceType)
    return clusters

def plot_clusters(filename, clusters, data, data_name):
    import matplotlib.pyplot as plt
    x, y, c = zip(*([(data.values[p][0], data.values[p][1], label) for label, cl in enumerate(clusters) for p in cl]))
    plt.style.use('ggplot')
    plt.clf()
    plt.title('DBSCAN Clustring on {}'.format(data_name))
    plt.scatter(x, y, c=c)
    print("-> writing cluster plot to {}".format(filename))
    plt.savefig(filename)

def plot_clusters_3d(filename, clusters, data, data_name):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    x, y, z, c = zip(*([(data.values[p][0], data.values[p][1], data.values[p][2], label) for label, cl in enumerate(clusters) for p in cl]))
    plt.style.use('ggplot')
    plt.clf()
    plt.title('Heirarchical Clustering on {}'.format(data_name))
    ax = Axes3D(plt.figure())
    ax.scatter(x, y, z, c=c)
    print("-> writing cluster plot to {}".format(filename))
    plt.savefig(filename)

# 3d plot of e vs numpoints vs variance
def plot_stats(filename, data, data_name):
   import matplotlib.pyplot as plt
   from mpl_toolkits.mplot3d import Axes3D
   es = []
   numpoints = []
   variances = []
   sizes = []
   e_step = .5
   min_var = float('inf')
   best_combo = (None,None)
   for numPoints in range(2,11): #numpoints
      e = 1
      while(e <= 10):
         clusters = dbscan(data,numPoints,e,"manhattan")
         stats = cluster_stats(clusters,data)
         num_clusters = len(stats)
         avg_var = sum([cluster["Dist-Variance"] for _, cluster in stats.items()]) / float(num_clusters)
         
         if(avg_var < min_var):
            min_var = avg_var
            best_combo = (e,numPoints)

         es.append(e)
         numpoints.append(numPoints)
         variances.append(avg_var)
         sizes.append(num_clusters)
         e+=e_step
   plt.style.use('ggplot')
   plt.clf()
   plt.title("DBSCAN on {}".format(data_name))
   ax = Axes3D(plt.figure())
   ax.set(xlabel="epsilon",ylabel="numPoints",zlabel="variance")
   ax.scatter(es,numpoints,variances)
   print("-> writing stats plot to  {}".format(filename))
   print("Best Variance = {}, achieved with e={}, numPoints={}".format(min_var,best_combo[0],best_combo[1]))
   plt.savefig(filename)
   '''fig, (ax1, ax2) = plt.subplots(2, sharex=True)
   fig.suptitle('Heirarchical Clustering on {}'.format(data_name))
   fig.text(0.9, 0.9, "link method: " + link_method, fontsize=12)
   # Variance
   ax1.plot(thresholds, variances)
   ax1.set(ylabel="Avg. variance dist-to-center")
   # Size
   ax2.plot(thresholds, sizes)
   ax2.set(ylabel="Number of clusters", xlabel="threshold")
   print("-> writing stats plot to {}".format(filename))
   plt.savefig(filename)'''


if __name__ == "__main__":
   parser = ArgumentParser(description="DBSCAN clustering.")
   parser.add_argument("filename", help="Name of the input file in csv form. Note that the first line of this file must be a restrictions vector.")
   parser.add_argument("--e",type=float, help="epsilon value for size of the sphere used when computing clusters.")
   parser.add_argument("--numpoints",type=int, help="The number of points to constitute suitable density for a cluster.")
   parser.add_argument("--headers", help="Name of input file containing header names on a single line in CSV format. Ommitting this will produce plots with unnamed axis")
   parser.add_argument("--plot-stats", action="store_true", help="Plot various statistics against varying threshold values")
   parser.add_argument("--plot-clusters", action="store_true", help="Plot clusters generated on 2-d data (note that this works only on 2-d data)")
   args = parser.parse_args()
   if not args.e:
      raise Exception("Must provide an float value for epsilon")
   if not args.numpoints:
      raise Exception("Must provide an integer value for numpoints")
   headers = None
   if args.headers:
      headers = parse_header(args.headers)
   data = parse_data(args.filename,headers)
   clusters = dbscan(data,args.numpoints,args.e,"manhattan")
   stats = cluster_stats(clusters,data)
   for name, stat in stats.items():
      print(name)
      for k,v in stat.items():
         print("  {}: {}".format(k, v))
      print()
   timestamp = "_" + str(datetime.now().replace(microsecond=0)).replace(' ', '_').replace(':', '-')
   if args.plot_stats:
      plot_filename = path.basename(args.filename).split('.')[0] + "_stats" + timestamp + ".png"
      plot_stats(plot_filename, data, args.filename)
   if args.plot_clusters:
      plot_filename = path.basename(args.filename).split(".")[0] + "_clusters"+timestamp+".png"
      if len(data.columns) == 2:
         plot_clusters(plot_filename,clusters,data,args.filename)
      elif len(data.columns) == 3:
         plot_clusters_3d(plot_filename,clusters,data,args.filename)
      else:
         raise Exception("Cannot plot data with {} dimensions,must be 2".format(len(data.columns)))


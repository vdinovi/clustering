import numpy as np
import pandas as pd
from parser import *
from random import randint
from math import sqrt
from stats import cluster_stats
from argparse import ArgumentParser
import pdb

def getTestData():
   return parse_data("../dataset/4clusters.csv")

def getInitialCentroids(data, k):
   #random centroids
   maxs = [np.max(data[i]) for i in data]
   mins = [np.min(data[i]) for i in data]
   d = {i:[ randint(mins[i],maxs[i]+1) for key in range(k)] for i in range(len(maxs))}
   df = pd.DataFrame(data=d)
   return df

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

if __name__ == "__main__":
  parser = ArgumentParser(description="kmeans clustering.")
  parser.add_argument("filename", help="Name of the input file in csv form. Note that the first line of this file must be a restrictions vector.")
  parser.add_argument("--threshold",type=float, help="threshold for difference between clusters to end computation.")
  parser.add_argument("--k",type=int, help="number of clusters to find.")
  args = parser.parse_args()
  data = parse_data(args.filename)
  clusters = kmeans(data,args.k,"manhattan",args.threshold)
  stats = cluster_stats(clusters,data)
  for name, stat in stats.items():
     print(name)
     for k,v in stat.items():
        print("  {}: {}".format(k, v))
     print()

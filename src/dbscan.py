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
   if not labels[i]:
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
   labels = np.full(len(data),None)
   for i in range(len(core)):
      if(addLabel(core[i],labels,cluster)):
         densityConnected(data,core[i],e,distanceType,cluster,core,labels)
         cluster+=1
   noise = [i for i in range(len(labels)) if not labels[i]]
   border = [i for i in range(len(labels)) if (i not in noise) and (i not in core)]
   clusters = [[i for i in range(len(labels)) if labels[i] == j] for j in set(labels)]
   return noise, border, core, clusters
   
def dbscan(data,numPoints,e,distanceType):
    noise, border, core, clusters = dbscanHelper(data,numPoints,e,distanceType)
    return clusters


if __name__ == "__main__":
  parser = ArgumentParser(description="DBSCAN clustering.")
  parser.add_argument("filename", help="Name of the input file in csv form. Note that the first line of this file must be a restrictions vector.")
  parser.add_argument("--e",type=float, help="epsilon value for size of the sphere used when computing clusters.")
  parser.add_argument("--numpoints",type=int, help="The number of points to constitute suitable density for a cluster.")
  args = parser.parse_args()
  data = parse_data(args.filename)
  if not args.e:
      raise Exception("Must provide an float value for epsilon")
  if not args.numpoints:
      raise Exception("Must provide an integer value for numpoints")


  clusters = dbscan(data,args.numpoints,args.e,"manhattan")
  stats = cluster_stats(clusters,data)

  for name, stat in stats.items():
     print(name)
     for k,v in stat.items():
        print("  {}: {}".format(k, v))
     print()

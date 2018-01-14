import sys
import csv
import numpy as np
import igraph as gp
from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial
import pandas as pd

if len(sys.argv) != 2:
    print("Incorrect Format. Format - python sac1.py <0 | 0.5 | 1>")
    exit(1)
alpha = float(sys.argv[1])

attributes = pd.read_csv("data/fb_caltech_small_attrlist.csv")
#g.add_vertices(attributes.index)
#g = gp.Graph.Read_Ncol('data/fb_caltech_small_edgelist.txt', directed=False)
g = gp.Graph.Read_Edgelist(open('data/fb_caltech_small_edgelist.txt',"r"), directed=False)
#g.add_vertices(attributes.index)
#print g
cols=list(attributes)
print g
#g.add_vertices(cols)

for node in range(0,324):
        g.vs[node]["attributes"] = list(attributes.iloc[node])
        
membership_list=[]


def similarity(g):
        numofiterations=g.vcount();
        matrix=[[0]*(numofiterations) for _ in range(0,numofiterations)]
        for i in range(0,numofiterations):
                for j in range(0,numofiterations):
                        x=g.vs[i]["attributes"]
                        y=g.vs[j]["attributes"]
                        matrix[i][j]=1- spatial.distance.cosine(x,y)
        return matrix

initialsimilarity=similarity(g)
numofedges=g.ecount()
numofvertices=g.vcount()
communities=[]

for i in range(0,g.vcount()):
	communities.append([i])
	membership_list.append(i)

def belongstocommunity(node):
        for n,c in enumerate(communities):
                if node in c:
                        return n
        return None

for i in range(0,g.vcount()):
        communities.append([g.vs[i]])
        g.vs[i]["community"]=[i]

def QNewman(C):
        """sumG=0
        sumD=0
        totaledges=0
        for i in C:
            for j in C:
                if i!=j:
                    print "i"+str(i)
                    print "j"+str(j)
                    index_i=g.vs[i].index
                    index_j=g.vs[j].index
                    sumG=sumG+g[i,j]
                    sumD=g.degree(i)*g.degree(j)
        totaledges=totaledges+g.degree(i)
        #print "ppp->"+str(g.degree(i))
        #print totaledges
        Q_newman=(sumG-(sumD/totaledges))/totaledges"""
        #print membership_list
        initial_val = g.modularity(membership_list)
        new_val = g.modularity(C)        
        return new_val-initial_val
    
def Qattr(i,C):
    similarity_matrix=similarity(g)
    sim=0;
    for j in C:
        if i!=j:
            sim=sim+similarity_matrix[i][j]
    #print sim
    return sim/len(C)

def modularity(i,membership,Community):
    return ((alpha*QNewman(membership))+((1-alpha)*Qattr(i,Community)))

for x in range(0,15):
    print("phase"+str(x))
    for i in range(0,g.vcount()):
        gain=[]
        maximum=0;
        new_comm=belongstocommunity(i)
        for j in range(0,g.vcount()):
            if i!=j:
                new_membership=membership_list
                new_membership[i]=j
                new_community=g.vs[j]["community"]
                new_community.append(i)
                if modularity(i,new_membership,new_community)>maximum:
                    print "inside";
                    new_comm=j
                    maximum=modularity(new_membership)
        g.vs[i]["community"]=new_community
        g.vs[new_comm]["community"].append(i)
        g.vs[i]["community"]=g.vs[new_comm]["community"]
        communities[i].remove(i)
        communitites[j].add(i)
        membership_list[i]=new_comm
    print x

communities = [z for z in communities if z!=[]]
new_communities = list(communities)
for c in new_communities:
                    print c;


        
            
    

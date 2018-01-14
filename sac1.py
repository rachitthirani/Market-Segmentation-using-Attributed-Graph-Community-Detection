import sys
import csv
import numpy as np
import igraph as gp
from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial
import pandas as pd
import math

if len(sys.argv) != 2:
    print("Incorrect Format. Format - python sac1.py <0 | 0.5 | 1>")
    exit(1)
alpha = float(sys.argv[1])

attributes = pd.read_csv("data/fb_caltech_small_attrlist.csv")
g = gp.Graph.Read_Edgelist(open('data/fb_caltech_small_edgelist.txt',"r"), directed=False)
cols=list(attributes)
for i in range(len(attributes)):
    for j in range(len(cols)):
        g.vs[i][cols[j]]=attributes.iloc[i][j]

membership_list=[]

def dot_product(v1, v2):
    return sum(map(lambda x: x[0] * x[1], zip(v1, v2)))

def cosineSimilarity(v1, v2):
    prod = dot_product(v1, v2)
    len1 = math.sqrt(dot_product(v1, v1))
    len2 = math.sqrt(dot_product(v2, v2))
    return prod / (len1 * len2)

def similarity(g):
    numofiterations=g.vcount();
    matrix=[[0]*(numofiterations) for _ in range(0,numofiterations)]
    for i in range(0,numofiterations):
        for j in range(0,numofiterations):
             x=g.vs[i].attributes().values()
             y=g.vs[j].attributes().values()
             matrix[i][j]=cosineSimilarity(x,y)
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

print("length"+str(len(communities)))
print(g.vcount())

def QNewman(C):
    initial_val = g.modularity(membership_list)
    new_val = g.modularity(C)
    return new_val-initial_val
    
def Qattr(i,C):
    sim=0;
    for j in C:
        if i!=j:
            sim=sim+initialsimilarity[i][j]
    return sim/len(C)

def modularity(i,membership,Community):
    return ((alpha*QNewman(membership))+((1-alpha)*Qattr(i,Community)))

new_membership=list(membership_list)

for x in range(0,15):
    print("iteration"+str(x))
    for i in range(0,g.vcount()):
        gain=[]
        maximum=0.0;
        new_comm=belongstocommunity(i)
        new_maximum=0.0
        new_community=[]
        for j in range(len(communities)):
            new_new_membership=list(new_membership)
            new_community=list(communities[j])
            if(i not in new_community):
                new_community.append(i)
                new_new_membership[i]=new_community[0]
            new_modularity=modularity(i,new_new_membership,new_community)
            if new_modularity>new_maximum:
                new_comm=j
                new_maximum=new_modularity
                maximum=new_maximum
        if(i not in communities[new_comm]):
            communities[belongstocommunity(i)].remove(i)
            communities[new_comm].append(i)
        new_membership[i]=new_comm
        membership_list=list(new_membership)
    communities = [z for z in communities if z!=[]]
    print x

communities = [z for z in communities if z!=[]]
new_communities = list(communities)

phase1_communities=list(communities)
phase1_membership=list(membership_list)

g.contract_vertices(new_membership,combine_attrs="mean")
g.simplify(multiple = True, loops = True, combine_edges = sum)

membershiplist=[]
newsimilarity=similarity(g)
communities=[]
for i in range(g.vcount()):
	communities.append([i])
	membership_list.append(i)

for x in range(0,15):
    print("iteration"+str(x))
    for i in range(0,g.vcount()):
        gain=[]
        maximum=0.0;
        new_comm=belongstocommunity(i)
        new_maximum=0.0
        new_community=[]
        for j in range(len(communities)):
            new_new_membership=list(new_membership)
            new_community=list(communities[j])
            if(i not in new_community):
                new_community.append(i)
                new_new_membership[i]=new_community[0]
            new_modularity=modularity(i,new_new_membership,new_community)
            if new_modularity>new_maximum:
                new_comm=j
                new_maximum=new_modularity
                maximum=new_maximum
        if(i not in communities[new_comm]):
            communities[belongstocommunity(i)].remove(i)
            communities[new_comm].append(i)
        new_membership[i]=new_comm
        membership_list=list(new_membership)
    communities = [z for z in communities if z!=[]]

communities = [z for z in communities if z!=[]]
new_communities = list(communities)
phase2_communities=list(new_communities)
i=0
newlist=[]
for c in phase2_communities:
     newlist.append([])
     for cc in c:
         for ccc in phase1_communities[cc]:
             newlist[i].append(ccc)
     i=i+1
print("community leangth= " + str(len(communities)))
with open('communities.txt', 'w') as data:
    for c in newlist:
        str_comm = ""
	for nodes in c:
            str_comm += str(nodes)
            str_comm += ", "
	data.write(str_comm)
	data.write("\n")
	










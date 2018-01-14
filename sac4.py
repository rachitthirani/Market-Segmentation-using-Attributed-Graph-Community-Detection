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
#g.add_vertices(attributes.index)
#g = gp.Graph.Read_Ncol('data/fb_caltech_small_edgelist.txt', directed=False)
g = gp.Graph.Read_Edgelist(open('data/fb_caltech_small_edgelist.txt',"r"), directed=False)
#g.add_vertices(attributes.index)
#print g
cols=list(attributes)
#print g
#g.add_vertices(cols)

#for node in range(0,324):
    #g.vs[node]["attributes"] = list(attributes.iloc[node])
#   g.vs[node][atttributes] = list(attributes.iloc[node])
"""
file_attrlist = open("data/fb_caltech_small_attrlist.csv", 'r')
attrList = {}
header = csv.Sniffer().has_header(file_attrlist.read(1024))
file_attrlist.seek(0)
attrData = csv.reader(file_attrlist)
if(header):
  attrNames = next(attrData)
for temp in attrNames:
	attrList[temp] = []
for entry in attrData:
	index = 0
	for at in entry:
		attrList[attrNames[index]].append(at)
		index += 1;
print attrList
for attribute in attrList:
	g.vs[attribute] = list(map(int, attrList[attribute]))
print g.vs[attribute]

file_attrlist.close()
"""
for i in range(len(attributes)):
    #print len(attributes)
    for j in range(len(cols)):
        g.vs[i][cols[j]]=attributes.iloc[i][j]
        #print cols[j]
        #print attributes.iloc[i][j]
        #print g.vs[i][cols[j]]
        #raw_input("length")




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
#            x=g.vs[i]["attributes"]
#            y=g.vs[j]["attributes"]
             x=g.vs[i].attributes().values()
             y=g.vs[j].attributes().values()
             #print x
             #print y
             #raw_input("x,y")
             matrix[i][j]=cosineSimilarity(x,y)
    """for i in range(g.vcount()):
		for j in range(g.vcount()):
			attr_i = list(g.vs[i].attributes().values())
			attr_j = list(g.vs[j].attributes().values())
			matrix[i][j] = cosineSimilarity(attr_i, attr_j)"""
    #print matrix
    #raw_input("matrix")

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
    #print new_val
    #print initial_val
    #raw_input("hello")
    return new_val-initial_val
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


def Qattr(i,C):
    sim=0;
    for j in C:
        if i!=j:
            sim=sim+initialsimilarity[i][j]
    #print sim
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
                #print ("here")
                new_community.append(i)
                #print new_community
                new_new_membership[i]=new_community[0]
            new_modularity=modularity(i,new_new_membership,new_community)
            if new_modularity>new_maximum:
                    #print ("inside")
                new_comm=j
                #print new_comm
                new_maximum=new_modularity
                #print new_maximum
                #raw_input("modularity gain:")
                maximum=new_maximum
        if(i not in communities[new_comm]):
            #print("here. its new")
            #print communities[belongstocommunity(0)]
            #print communities[new_comm]
            communities[belongstocommunity(i)].remove(i)
            communities[new_comm].append(i)
#      g.vs[new_comm]["community"].append(i)
#        g.vs[i]["community"]=list(g.vs[new_comm]["community"])
#        communities.remove(communities[i])
#        communities[new_comm].append(i)
        #communities[i]=communities[new_comm]
        new_membership[i]=new_comm
        membership_list=list(new_new_membership)
    #print new_membership
        #print len(communities)
        #raw_input("community len")
    communities = [z for z in communities if z!=[]]
    #for c in range(len(communities)-1):
        #print c
        #communities[c]=new_membership[c];
        #print communities[c]
        #raw_input("new");
        #communities.remove(communities.index(c))
        #print g.vs[c]["community"];
    #raw_input("new membership")
    print x

communities = [z for z in communities if z!=[]]
new_communities = list(communities)
for c in new_communities:
    print c;
print("community leangth= " + str(len(communities)))

nodecount=0
new_node=[0]*324
print "this is new membership"
print new_membership

for c in communities:
	for node in c:
		new_node[node] = nodecount
	nodecount += 1

#g.contract_vertices(new_node, combine_attrs = "mean")
print "attributes"
print g.vs[1].attributes().values()
phase1_communities=list(communities)
phase1_membership=membership_list
print phase1_membership
raw_input("phase1 membership")
print phase1_communities
raw_input("phase1 community")
g.contract_vertices(new_membership,combine_attrs="mean")


g.simplify(multiple = True, loops = True, combine_edges = sum)
print "after contating"
print g

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
                #print ("here")
                new_community.append(i)
                #print new_community
                new_new_membership[i]=new_community[0]
            new_modularity=modularity(i,new_new_membership,new_community)
            if new_modularity>new_maximum:
                    #print ("inside")
                new_comm=j
                #print new_comm
                new_maximum=new_modularity
                #print new_maximum
                #raw_input("modularity gain:")
                maximum=new_maximum
        if(i not in communities[new_comm]):
            #print("here. its new")
            #print communities[belongstocommunity(0)]
            #print communities[new_comm]
            communities[belongstocommunity(i)].remove(i)
            communities[new_comm].append(i)
#      g.vs[new_comm]["community"].append(i)
#        g.vs[i]["community"]=list(g.vs[new_comm]["community"])
#        communities.remove(communities[i])
#        communities[new_comm].append(i)
        #communities[i]=communities[new_comm]
        new_membership[i]=new_comm
        membership_list=list(new_new_membership)
    #print new_membership
        #print len(communities)
        #raw_input("community len")
    communities = [z for z in communities if z!=[]]
    #for c in range(len(communities)-1):
        #print c
        #communities[c]=new_membership[c];
        #print communities[c]
        #raw_input("new");
        #communities.remove(communities.index(c))
        #print g.vs[c]["community"];
    #raw_input("new membership")
    print x

communities = [z for z in communities if z!=[]]
new_communities = list(communities)
phase2_communities=list(new_communities)
print phase2_communities
raw_input("phase2_communities")
i=0
newlist=[]
for c in phase2_communities:
     newlist.append([])
     for cc in c:
         for ccc in phase1_communities[cc]:
             newlist[i].append(ccc)
     i=i+1
print newlist
raw_input("newlist")
         
print "new communities"
print new_communities
for c in new_communities:
    print c;
print("community leangth= " + str(len(communities)))
with open('communities.txt', 'w') as op:
    for c in newlist:
        str_comm = ""
	for nodes in c:
            str_comm += str(nodes)
            str_comm += ", "
            print str_comm
	op.write(str_comm)
	op.write("\n")
	










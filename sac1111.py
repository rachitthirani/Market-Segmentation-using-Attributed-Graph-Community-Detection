#Take one argument as input - Alpha value
#Output - communities.txt - One community per line - vertex ids separated by commas
#Output - communities_0.txt, communities_5.txt, communities_1.txt for alpha = 0,0.5,1
#Also submit plots generated for the above three by running evaluation.R
#Also submit the original data folder
#igraph package, cosine similarity
#SAC-1 Method from Section III and Sectiom IV Part A of paper
#Limit convergence to have a maximum of 15 iterations

### Topic 7 - Project 6 : Market Segmentation ###
# unity id - avshirod #
# due date - 3.20.2016 Sun #

from igraph import *
import csv
from scipy import spatial

'''
g=Graph()
g.add_vertices(3)
g.add_edges([(0,1),(1,2)])
g.get_eid(1,2)
g.delete_edges(1)
g.delete_vertices(1)
summary(g)
print(g)
g.get_edgelist()
g.vs["name"] = ["A"]
g.degree()
g.es[0].source, target, index, tuple, attributes()
g.vs[0].index, attributes()
[g.es[idx].tuple for idx, eb in enumerate(ebs) if eb == max_eb]
g.Read_EdgeList()
g.vcount()
g.ecount()
g.adjacent(0)
g.get_adjacency()
g.get_adjlist()
g.is_weighted()
g.modularity(membership_list)
'''

g = Graph()
g.add_vertices(324)

curr_dir = os.getcwd()
data_path = curr_dir + "\\data\\"
filepath_edgelist = data_path + "fb_caltech_small_edgelist.txt"
filepath_attrlist = data_path + "fb_caltech_small_attrlist.csv"
file_edgelist = open("data/fb_caltech_small_edgelist.txt", 'r')
file_attrlist = open("data/fb_caltech_small_attrlist.csv", 'r')

if len(sys.argv) != 2:
    print("Please Use Format - python sac1.py <0 | 0.5 | 1>")
    exit(1)
alpha_value = float(sys.argv[1])

# Add Edges
# g.Read_Edgelist() should've worked; but didn't
file_edgelist.seek(0)
for edge in file_edgelist:
	edge = edge.strip()
	if edge:
		start, end = edge.split(' ', 1)
		g.add_edge(int(start), int(end))

# Add Attributes
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
for attribute in attrList:
	g.vs[attribute] = list(map(int, attrList[attribute]))

file_attrlist.close()
file_edgelist.close()


### ----------------  Phase 1  ----------------- ###

# Similarity Matrix
def getSimilarityMatrix(g):
	noOfVertices = g.vcount()
	simA = [[0]*noOfVertices for _ in range(noOfVertices)]
	for i in range(noOfVertices):
		for j in range(noOfVertices):
			attr_i = list(g.vs[i].attributes().values())
			attr_j = list(g.vs[j].attributes().values())
			simA[i][j] = cosineSimilarity(attr_i, attr_j)
	return simA

'''
def cosineSimilarity(vec1, vec2):
	return (1-spatial.distance.cosine(vec1, vec2))
'''

import math
# from itertools import zip

def dot_product(v1, v2):
    return sum(map(lambda x: x[0] * x[1], zip(v1, v2)))

def cosineSimilarity(v1, v2):
    prod = dot_product(v1, v2)
    len1 = math.sqrt(dot_product(v1, v1))
    len2 = math.sqrt(dot_product(v2, v2))
    return prod / (len1 * len2)

# cosine = format(round(cosineSimilarity(v1, v2), 3))

simA = getSimilarityMatrix(g)

noOfVertices = g.vcount()
noOfEdges = g.ecount()
communities = []
noOfCommunities = noOfVertices

# Initialize each node to a separate community
for i in range(noOfVertices):
	communities.append([i])

def getCommunity(nodeNum):
	return next((comm_num for comm_num, comm1 in enumerate(communities) if nodeNum in comm1), None)

# Community attribute for each node indicates which community it belongs to
communityList = []
for i in range(noOfVertices):
	communityList.append(getCommunity(i))
g.vs["community"] = communityList

# Moving node x to Community C
def composite_modularity_gain(x, C):
	delta_Q_Newman = Q_Newman(x, C)
	delta_Q_Attr = Q_Attr(x, C)
	delta_Q = alpha_value * delta_Q_Newman + (1 - alpha_value) * delta_Q_Attr
	return delta_Q

def Q_Newman(x, C):
	adjListX = g.get_adjlist()[x]
	#print adjListX
	#print C
	#print x
	sumGix = sum([1 for node in communities[C] if node in adjListX])
	sumdi = sum([g.degree(node) for node in communities[C]])
	dx = g.degree(x)
	return ((sumGix - ((dx * sumdi) / (2 * noOfEdges))) / (2 * noOfEdges))

def Q_Attr(x, C):
	return (sum([(simA[x][node]) for node in communities[C]])) / ((len(communities)*len(communities)))

def printComm():
	for c in communities:
		if not c:
			print(', '.join(map(str, c[:])))

for iteration in range(15):
	#print("Running Itr " + str(iteration))
	for i in range(noOfVertices):
		cmg = []
		for j in range(noOfVertices):
			comm_i = g.vs[i]["community"]
			comm_j = g.vs[j]["community"]
			g.vs[i]["community"] = comm_j
			cmg.append(composite_modularity_gain(i, comm_j))
		if max(cmg) > 0:
			communities[getCommunity(i)].remove(i)
			communities[cmg.index(max(cmg))].append(i)
			g.vs[i]["community"] = cmg.index(max(cmg))
		else: g.vs[i]["community"] = comm_i
	#printComm()
	# communities = filter(None, communities)

communities = [z for z in communities if z!=[]]
communities_phase1 = list(communities)
print communities_phase1

### ----------------  Phase 2  ----------------- ###

new_nodes = [0]*noOfVertices

nodeCount = 0
for c in communities:
	for node in c:
		new_nodes[node] = nodeCount
	nodeCount += 1

g.contract_vertices(new_nodes, combine_attrs = mean)
g.simplify(multiple = True, loops = True, combine_edges = sum)

simA = getSimilarityMatrix(g)

noOfVertices = g.vcount()
noOfEdges = g.ecount()
communities = []
noOfCommunities = noOfVertices

# Initialize each node to a separate community
for i in range(noOfVertices):
	communities.append([i])

# Community attribute for each node indicates which community it belongs to
communityList = []
for i in range(noOfVertices):
	communityList.append(getCommunity(i))
g.vs["community"] = communityList

for iteration in range(15):
	#print("Running Itr " + str(iteration))
	for i in range(noOfVertices):
		cmg = []
		for j in range(noOfVertices):
			comm_i = g.vs[i]["community"]
			comm_j = g.vs[j]["community"]
			g.vs[i]["community"] = comm_j
			cmg.append(composite_modularity_gain(i, comm_j))
		if max(cmg) > 0:
			communities[getCommunity(i)].remove(i)
			communities[cmg.index(max(cmg))].append(i)
			g.vs[i]["community"] = cmg.index(max(cmg))
		else: g.vs[i]["community"] = comm_i
	#printComm()
	# communities = filter(None, communities)

communities = [z for z in communities if z]
with open('communities.txt', 'w') as op:
	for c in communities:
		if c:
			str_comm = ""
			for nodes in c:
				str_comm += str(communities_phase1[nodes])[1:-1]
				str_comm += ", "
				#op.write(', '.join(map(str, communities_phase1[nodes])))
				#op.write('&')
			str_comm = str_comm[:-2]
			op.write(str_comm)
			op.write("\n")

'''
import math
def cosine_similarity(v1,v2):
    "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]; y = v2[i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y
    return sumxy/math.sqrt(sumxx*sumyy)
'''

'''
Resources:
http://igraph.org/python/doc/igraph-module.html
http://igraph.org/python/doc/tutorial/tutorial.html
http://igraph.org/python/doc/igraph.GraphBase-class.html
http://igraph.org/python/doc/igraph.Graph-class.html#modularity

'''

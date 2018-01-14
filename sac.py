import sys
import csv
import numpy as np
import igraph as gp
from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial
import pandas as pd


g = gp.Graph()
attributes = pd.read_csv("data/fb_caltech_small_attrlist.csv")
#g.add_vertices(attributes.index)
g = gp.Graph.Read_Ncol('data/fb_caltech_small_edgelist.txt', directed=False)
#g.add_vertices(attributes.index)
#print g
cols=list(attributes)
#g.add_vertices(cols)

for node in range(0,324):
        g.vs.find(name=str(node))["attributes"] = list(attributes.iloc[node])



def similarity(g):
        numofiterations=g.vcount();
        matrix=[[0]*numofiterations for _ in range(numofiterations)]
        for i in range(0,numofiterations):
                for j in range(0,numofiterations):
                        x=g.vs.find(name=str(i))["attributes"]
                        y=g.vs.find(name=str(j))["attributes"]
                        matrix[i][j]=1- spatial.distance.cosine(x,y)
        return matrix

initialsimilarity=similarity(g)
numofedges=g.ecount()
numofvertices=g.vcount()
communities=[]

for i in range(g.vcount()):
	communities.append([i])

def belongstocommunity(node):
        for n,c in enumerate(communities):
                if node in c:
                        return n
        return None

for i in range(0,g.vcount()):
        communities.append([g.vs.find(name=str(i))])
        g.vs.find(name=str(i))["community"]=[i]
        print g.vs.find(name=str(i))["community"]


def QNewman(x, C):
	adjListX = g.get_adjlist()[x]
	print adjListX
	print C
	print x
	sumGix = sum([1 for node in communities[C] if node in adjListX])
	sumdi = sum([g.degree(node) for node in communities[C]])
	dx = g.degree(x)
	return ((sumGix - ((dx * sumdi) / (2 * noOfEdges))) / (2 * noOfEdges))

def QAttr(x, C):
	return (sum([(simA[x][node]) for node in communities[C]])) / ((len(communities)*len(communities)))

        
def modularity_gain(x, C):
	Q_Newman = QNewman(x, C)
	Q_Attr = QAttr(x, C)
	delta_Q = alpha_value * Q_Newman + (1 - alpha_value) * Q_Attr
	return delta_Q

def displacommunity():
	for c in communities:
		if not c:
			print(', '.join(map(str, c[:])))

for iteration in range(0,15):
	print("Running Itr " + str(iteration))
	for i in range(0,g.vcount()):
		cmg = []
		for j in range(0,g.vcount()):
			comm_i = g.vs.find(str(i))["community"]
			print comm_i
			comm_j = g.vs.find(str(j))["community"]
			print comm_j
			g.vs.find(str(i))["community"] = comm_j
			cmg.append(modularity_gain(i, comm_j))
		if max(cmg) > 0:
			communities[belongstocommunity(i)].remove(i)
			communities[cmg.index(max(cmg))].append(i)
			g.vs.find(str(i))["community"] = cmg.index(max(cmg))
		else: g.vs.find(str(i))["community"] = comm_i
	printComm()
	#communities = filter(None, communities)

communities = [z for z in communities if z!=[]]
communities_phase1 = list(communities)


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
for i in range(numofvertices):
	communities.append([i])

# Community attribute for each node indicates which community it belongs to
communityList = []
for i in range(numofvertices):
	communityList.append(getCommunity(i))
g.vs["community"] = communityList

for iteration in range(15):
	#print("Running Itr " + str(iteration))
	for i in range(numofvertices):
		cmg = []
		for j in range(numofertices):
			comm_i = g.vs.find(str(i))["community"]
			comm_j = g.vs.find(str(j))["community"]
			g.vs.find(str(i))["community"] = comm_j
			cmg.append(composite_modularity_gain(i, comm_j))
		if max(cmg) > 0:
			communities[getCommunity(i)].remove(i)
			communities[cmg.index(max(cmg))].append(i)
			g.vs.find(str(i))["community"] = cmg.index(max(cmg))
		else: g.vs.find(str(i))["community"] = comm_i
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







        
"""if len(sys.argv) != 2:
    print("Incorrect Format. Format - python sac1.py <0 | 0.5 | 1>")
    exit(1)
alpha = float(sys.argv[1])"""





"""
g = gp.Graph()
g.add_vertices(324)

#curr_dir = os.getcwd()
#data_path = curr_dir + "\\data\\"
filepath_edgelist =  "data/fb_caltech_small_edgelist.txt"
filepath_attrlist = "data/fb_caltech_small_attrlist.csv"
file_edgelist = open(filepath_edgelist, 'r')
file_attrlist = open(filepath_attrlist, 'r')

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

print g

"""










"""def output_to_file(clusters, filename):
	f = open(filename,'w')
	data = "\n".join(",".join(str(x) for x in clusters[item]) for item in clusters.iterkeys())
	f.write(data)	
	f.close()

def make_graph:
    attributes = pd.read_csv("data/fb_caltech_small_attrlist.csv")
    edges = gp.Graph.Read_Edgelist('data/fb_caltech_small_edgelist.txt', directed=False)

    cols = attributes.columns.tolist()
    
    for i, item in sac_attributes.iterrows():
    	attributes.vs[i]["attribute"] = item

    attributes.es["weight"] = 1
    attributes.vs["cardinality"] = 1
    return attributes    
    
def clustering():
    graph = make_graph()

def sac(graph,silimarity_matrix):
    num_nodes = len(sac_graph.vs)
    for i in range(0,num_nodes):
        for j in range(0,num_nodes):
            (1 - spatial.distance.cosine(dataSetI, dataSetII))"""
            
    


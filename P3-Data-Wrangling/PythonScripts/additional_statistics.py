
# coding: utf-8

# In[1]:

from pymongo import MongoClient
import pprint


# In[2]:

def get_db():
    client = MongoClient('localhost:27017')
    db = client.OSM
    return db


# In[3]:

db = get_db()


# In[5]:

schools_dict = {}

# Filter for all schools with a name
schools = db.bangalore.aggregate([
    {"$match" : {"amenity": "school",
                    "name" : {"$exists": 1}}
    }
])

for sc in schools:
    # Nodes have Pos data, so we take the nodes and add the Pos information to a dictionary with school name as key
    if sc["type"] == "node":
        schools_dict[sc["name"]] = sc["pos"]
    # Ways do not have Pos data. But they have a node reference and the that in turn can give us Pos data
    else:
        # Collect all node_refs with the school names
        schools_node_db = db.bangalore.aggregate([
           {
            "$match" : {"amenity": "school",
            "name" : {"$exists": 1}}
           },
           {
            "$match" : {"node_refs" : {"$exists" : 1}}
           },
           {
            "$project" : {"name" : "$name", "node" : "$node_refs"}
           }
        ])
        schools = {}
        # Loop thorugh the pipeline result object and add the node_refs to a list in a dictionary with school names as keys
        for school in schools_node_db:
            n = school["node"]
            values = []
            for i in n:
                values.append(i)
            schools[school["name"]] = values      


# In[10]:

# Here we iterate through the dictionary and look for a matching node. Then we get the node Pos
for sc,value in schools.iteritems():
    sc_pos_db =  db.bangalore.aggregate([
    {
    "$match" : {"type" : "node",
               "id" : {"$in" : value}
                }
    },
    {"$project" : {"name" : sc, "pos" : "$pos", "_id" : 0}}
    ])
    # A school can have more than one node and each of these nodes give a slightly different Pos. In case of multiple values,
    # only one of them gets added to the dictionary
    for s in sc_pos_db:
        schools_dict[s["name"]] = s["pos"]

print len(schools_dict)


# In[7]:

# Collect all residential buildings information
buildings_node_db = db.bangalore.aggregate([
    {
    "$match" : {
                "$or" :
               [
                   {"building" : "Residential House"},
                   {"building" : "apartment"},
                   {"building" : "apartments"},
                   {"building" : "residential"},
                   {"building" : "house"}
               ]
    }
    },
    {
        "$match" : {"node_refs" : {"$exists" : 1}}
    },
    {"$project" : {"node" : "$node_refs"}}
])

# All buildings are in ways. So they do not have pos information. We take the node_refs information here and add them to a list
building_nodes = []
for b in buildings_node_db:
    n = b["node"]
    for i in n:
        building_nodes.append(i)
print len(building_nodes)


# In[8]:

# Find a matching node for each node_ref from the list
building_pos_db =  db.bangalore.aggregate([
    {
    "$match" : {"type" : "node",
               "id" : {"$in" : building_nodes}
                }
    }
])

# Get the latitude and longitude values for the buildings
building_pos = []
for i in building_pos_db:
    lat = i["pos"][0]
    lon = i["pos"][1]
    building_pos.append([lat,lon])
print len(building_pos)


# In[34]:

# Create a square shaped range around each school and look for number of houses in that area. The side of a square is 0.08.
houses_near_schools = {}
for school in schools_dict.keys():
    count=0
    lat = schools_dict[school][0]
    lon = schools_dict[school][1]
    for build in building_pos:
        if build[0] >= lat - 0.04 and         build[0] <= lat+0.04 and         build[1] >= lon - 0.04 and         build[1] <= lon + 0.04:
            count+=1
    houses_near_schools[school] = count
    
#Sort the dictionary keys based on values    
sorted_values= sorted(houses_near_schools, key = lambda x: houses_near_schools[x], reverse=True)

for school in sorted_values:
    print("There are {} houses near {}".format(houses_near_schools[school] , school))


# In[ ]:




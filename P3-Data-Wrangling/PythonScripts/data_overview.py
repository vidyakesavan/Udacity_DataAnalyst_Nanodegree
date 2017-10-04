
# coding: utf-8

# In[1]:

from pymongo import MongoClient
import pprint


# In[2]:

# Get a database connection
def get_db():
    client = MongoClient('localhost:27017')
    db = client.OSM
    return db


# In[3]:

db = get_db()


# In[4]:

# Count number of documents in the DB
db.bangalore.count()


# In[5]:

# Number of nodes and ways
pipeline = [{"$group" : {"_id" : "$type",
                    "count" : {"$sum" : 1}}}]
nw = db.bangalore.aggregate(pipeline)
for n in nw:
    print n


# In[6]:

# Top 10 users by contribution count
users = db.bangalore.aggregate([
    {"$group" : {"_id" : "$created.uid",
                    "count" : {"$sum" : 1}}
    },
    {"$sort" : {"count" : -1}
    },
    {"$limit" : 10}  
])
for user in users:
    print user


# In[8]:

# First user entry
entries = db.bangalore.aggregate([
    {"$sort" : {"created.timestamp" : 1}
    },
    {"$project" : {"_id":0,"created.user":1 ,"created.timestamp":1, "created.uid":1 }
    },
    {"$limit" : 1}
])
for entry in entries:
    print entry


# In[9]:

# Most common postal code
postcode = db.bangalore.aggregate([
    {"$match" : {"address.postcode" : {"$exists" : 1}}
    },
    {"$group" : {"_id" : "$address.postcode",
                    "count" : {"$sum" : 1}}
    },
    {"$sort" :{"count" : -1}
    },
    {"$limit" : 1}
])
for code in postcode:
    print code


# In[10]:

# Top 5 amenities
amenities = db.bangalore.aggregate([
    {"$match" : {"amenity" : {"$exists" : 1}}
    },
    {"$group" : {"_id" : "$amenity",
                    "count" : {"$sum" : 1}}
    },
    {"$sort" : {"count" : -1}
    },
    {"$limit" : 5}
])
for amenity in amenities:
    print amenity


# In[11]:

# Most popular cuisine
restaurants = db.bangalore.aggregate([
    {"$match" : {"amenity" : {"$eq" : "restaurant"}}
    },
    {"$match" : {"cuisine" : {"$exists" : 1}}
    },
    {"$group" : {"_id" : "$cuisine",
                    "count" : {"$sum" : 1}}
    },
    {"$sort" : {"count" : -1}
    },
    {"$limit" :5}
])
for cuisine in restaurants:
    print cuisine


# In[ ]:




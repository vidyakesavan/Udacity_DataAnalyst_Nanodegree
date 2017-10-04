
# coding: utf-8

# In[2]:

import xml.etree.cElementTree as ET
import pprint
import re
import codecs
import json
from collections import defaultdict


# In[ ]:

"""
Below are the data wrangling tasks performed by this code:

1. Process only 2 types of top level tags: "node" and "way"

2. All attributes of "node" and "way" should be turned into regular key/value pairs, except:
    - attributes in the CREATED array should be added under a key "created"
    - attributes for latitude and longitude should be added to a "pos" array,
      for use in geospacial indexing. Make sure the values inside "pos" array are floats
      and not strings. 
      
3. If the second level tag "k" value starts with "addr:", it should be added to a dictionary "address"

4. Remove "," at the end of street names

5. Postcodes must be of the format 5xxxxx where x is a digit. Remove all spaces, special characters and alphabets. 
If postcode is all digits but greater than 6 digits, leave the postcode as is.

6. Fix all phone numbers to follow the format +91 followed by number in one of these formats 
        +91xxxxxxxxxx (for all mobile numbers) 
        +91xxxxxxxxxxx (for landline numbers starting with city code 080 for Bangalore) 
        1800xxxxxxx (for toll free numbers)
        
7. If there is a second ":" that separates the type/direction of a street, the tag should be ignored

8. For "way" specifically:

      <nd ref="305896090"/>
      <nd ref="1719825889"/>

    should be turned into
    "node_refs": ["305896090", "1719825889"]

9. Wrangle the data into a list of dictionaries. The output should look like this:

{
"id": "1091213725",
"type: "node",
"visible":"true",
"created": {
          "version":"4",
          "changeset":"31340331",
          "timestamp":"2015-05-21T11:32:32Z",
          "user":"vamshiN",
          "uid":"2907738"
        },
"pos": [12.9673994, 77.7146901],
"address": {
          "housenumber": "125",
          "postcode": "560066",
          "street": "Brookfield Main Road"
        },
"amenity": "restaurant",
"cuisine": "indian;asian",
"name": "Hotel Zaica",
"phone": "+918041162485"
}

10. Website URLs must start with https:// or http://

11. House numbers must contain a digit and can end with a digit or one character. For example: 44, No. 44, 491/A

10. Convert the data into a json file and import into MongoDB with the following details:
    DB Name: OSM
    Collection: bangalore

"""


# In[3]:

#Regular expressions
add_re = re.compile(r'^\baddr:\b')
phone_number1 = re.compile(r'[^\s]\+91\d+')
phone_number2 = re.compile(r'\+91\d+')
website_re = re.compile(r'^\bhttp[s]?\b')
house_numbers_re = re.compile(r'\d+[/-]?[a-zA-Z]?$')

CREATED = ["version", "changeset", "timestamp", "user", "uid"]


# In[4]:

def process_map(file_in, pretty = False):
    file_out = "{0}.json".format(file_in)
    data = []
    with codecs.open(file_out, "w") as fo:
        for _, element in ET.iterparse(file_in):
            el = shape_element(element)
            if el:
                data.append(el)
                if pretty:
                    fo.write(json.dumps(el, indent=2)+"\n")
                else:
                    fo.write(json.dumps(el) + "\n")
    return data


# In[8]:

def shape_element(element):
    node = {}
    created = {}
    pos = []
    address = {}
    node_refs = []
    # Process only node and way
    if element.tag == "node" or element.tag == "way" :
        elem_list = element.attrib.keys()
        for elem in elem_list:
            # Add elements into created group
            if elem in CREATED:
                created[elem] = element.attrib[elem]
                node["created"] = created
            elif elem == "id" or elem == "visible":
                node[elem] = element.attrib[elem]
        
        # Convert latidude and longitude into float values and add to list
        if "lat" in element.keys() and "lon" in element.keys():
            pos = [float(element.attrib["lat"]),float(element.attrib["lon"])]
            node["pos"] = pos
    
        for tag in element.iter("tag"):
            k = tag.attrib["k"]
            # Continue with only elements with single ":"
            if not k.count(":") > 1:
                if re.search(add_re,k):
                    key = k.split(":")[1]
                    value = tag.attrib["v"]
                    # Remove "," from end of street names
                    if k == "addr:street":
                        value = value.strip(",")
                    # ensure that postcodes are only numbers
                    if k == "addr:postcode":
                        value = re.sub('[^0-9]','',value)
                    # house numbers must contain a digit
                    if k == "addr:housenumber":
                        value = check_house_number(value)
                    if(value):
                        address[key] = value
                    node["address"] = address
                else:
                    # for non address tags, replace ":" with "_"
                    key = k.replace(":","_")
                    value = tag.attrib["v"]
                    # Fix phone number to follow specified format
                    if k == "phone":
                        value = fix_phone_number(value)
                    # Check if websites start with http or https
                    if k == "website":
                        value = check_url(value)
                    if value:
                        node[key] = value
        # Add node_refs to a list
        for tag in element.iter("nd"):
            nd = tag.attrib["ref"]
            node_refs.append(nd)
        if node_refs:
            node["node_refs"] = node_refs               
        node["type"] = element.tag
        return node
    else:
        return None


# In[5]:

def fix_phone_number(phone):
    # If phone number is less than 10 digits, discard the number
    if sum(p.isdigit() for p in phone) < 10:
        return None
    
    m1 = phone_number1.match(phone)
    if m1 is None:
        # Remove : in phone
        if ":" in phone:
            phone = re.sub(":","",phone)
        # Remove () in phone
        if "(" in phone or ")" in phone:
            phone = re.sub("[()]","",phone)
        # Remove - in phone
        if "-" in phone:
            phone = re.sub("-","",phone)
        # Remove spaces in phone
        if " " in phone:
            phone = re.sub("\s+","",phone)
        # Remove quotes in phone
        if '"' in phone:
            phone = re.sub('"','',phone)
        # If phone starts with 91, append a + sign before the number
        if phone.startswith("91"):
            phone = "+" + phone
        # If phone starts with 0, remove 0
        if phone.startswith("0"):
            phone = phone[1:len(phone)]
        # If phone starts with 1800, do nothing and return as it is a toll free number
        if phone.startswith("1800"):
            return phone
    
        # At this stage, all phone numbers that already have a +91 should match the regular expression
        m2 = phone_number2.match(phone)
        # Add +91 to the ones that do not match
        if m2 is None:
            phone = "+91" + phone
    return phone


# In[6]:

# If the website URL starts with http or https, return the value. Else append http:// before the URL
def check_url(website):
    if re.search(website_re,website):
        return website
    else:
        return ("http://" + website)


# In[7]:

# Return the house number if it ends with a digit or a single alphabet
def check_house_number(house_number):
    if re.search(house_numbers_re,house_number):
        return house_number


# In[9]:

data = process_map('bengaluru_sample_k05.osm', False)


# In[ ]:




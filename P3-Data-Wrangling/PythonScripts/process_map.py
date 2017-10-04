
# coding: utf-8

# In[1]:

import xml.etree.cElementTree as ET
from collections import defaultdict
import re
import pprint


# In[2]:

## Sample file for Bangalore City Data
filename="bengaluru_sample_k05.osm"


# In[ ]:

postcode_good = set()
postcode_bad = set()

## Regular expression to check if the postcode starts with 5 and has a total of 6 numeric digits.
postcode_check = re.compile(r'\b^5\d{5}\b')

# For each postcode element, we classify it as good or bad postcode
for event, element in ET.iterparse(filename):
    if is_postcode(element):
        audit_postcode(postcode_good,postcode_bad,element.attrib["v"])

print postcode_bad


# In[4]:

## This fucntion checks if the element is a postcode element and returns the element is it is a postcode element
def is_postcode(elem):
    return (elem.tag == "tag") and (elem.attrib["k"] == "addr:postcode")


# In[5]:

## This function checks of the postcode matches the format for Bangalore. If yes, the postcode is added to the postcode_good set.
## Else, the postcode is added to the postcode_bad set
def audit_postcode(postcode_good,postcode_bad,postcode):
    if re.search(postcode_check,postcode):
        postcode_good.add(postcode)
    else:
        postcode_bad.add(postcode)


# In[10]:

## This is the regular expression to match the street type. We look for the last word in a string, it must start with an alphabet
## and can have an optional "." at the end
street_type_re = re.compile(r'\b\S+\.?$', re.IGNORECASE)

## These are the expected street types. 
EXPECTED = ["Road","Stage","Layout","Street","Nagar","Circle","Cross","Main","Block","Phase","Colony"]

## Go through each street element, collect all the street names and street types
all_streets = set()
street_types = defaultdict(set)
for event, element in ET.iterparse(filename):
    if is_street(element):
        collect_streetnames(all_streets,element.attrib["v"])
        audit_streetname(street_types,element.attrib["v"])

print len(all_streets)
print len(street_types)


# In[7]:

## This fucntion checks if the element is a street element and returns the element is it is a street element

def is_street(elem):
    return (elem.tag == "tag") and (elem.attrib["k"] == "addr:street")


# In[8]:

## This function just adds the street name to a set
def collect_streetnames(all_steeets,street_name):
    all_streets.add(street_name)


# In[9]:

## This function takes the street type from the regular expression output and adds it to a set
def audit_streetname(street_types,street_name):
    m = street_type_re.search(street_name)
    if m:
        street_type = m.group()
        if street_type.lower() not in (name.lower() for name in EXPECTED):
            street_types[street_type].add(street_name)


# In[11]:

print street_types.keys()


# In[5]:

all_phone = set()
for event, element in ET.iterparse(filename):
    if is_phone(element):
        collect_phonenumbers(all_phone,element.attrib["v"])
print len(all_phone)


# In[4]:

## This fucntion checks if the element is a phone element and returns the element is it is a phone element
def is_phone(elem):
    return (elem.tag == "tag") and (elem.attrib["k"] == "phone")


# In[3]:

## This function takes all phone numbers into a set
def collect_phonenumbers(all_phone,phone):
    all_phone.add(phone)


# In[6]:

#Print a small subset so we can see a few discrepancies
print list(all_phone)[100:120]


# In[73]:

# Website URLs must start with http or https 
website_re = re.compile(r'^\bhttp[s]?\b')

valid_url = []
invalid_url = []

# This function checks and returns a website element
def is_website(elem):
    return (elem.tag == "tag") and (elem.attrib["k"] == "website")

for event, element in ET.iterparse(filename):
    if is_website(element):
        check_url(element.attrib["v"])


# In[72]:

# This function categorises the websites as valid and invalid ones
def check_url(website):
    if re.search(website_re,website):
        valid_url.append(website)
    else:
        invalid_url.append(website)


# In[79]:

print "Valid URLS" , "\n" , valid_url[1:10],"\n"
print "Invalid URLS" , "\n" , invalid_url[1:10]


# In[40]:

## This fucntion checks if the element is a street element and returns the element is it is a street element

def is_housenumber(elem):
    return (elem.tag == "tag") and (elem.attrib["k"] == "addr:housenumber")


# In[80]:

# This function categorises the house numbers into valid and invalid ones

def check_house_number(house_number):
    if re.search(house_numbers_re,house_number):
        house_number_valid.append(house_number)
    else:
        house_number_invalid.append(house_number)    


# In[89]:

# house numbers must end with a digit or a single alphabet.
house_numbers_re = re.compile(r'\d+[/-]?[a-zA-Z]?$')

house_number_invalid = []
house_number_valid = []

for event, element in ET.iterparse(filename):
    if is_housenumber(element):
        check_for_number(element.attrib["v"])


# In[92]:

print "Valid House Numbers" , "\n" , house_number_valid[1:10],"\n"
print "Invalid House Numbers" , "\n" , house_number_invalid[1:10]


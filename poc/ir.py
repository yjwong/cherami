import json
import re
import HTMLParser

def remove_mentions(text):
    return re.sub('@([A-Za-z]+[A-Za-z0-9_]*)', '', text)

def remove_links(text):
    return re.sub(r'https?:\/\/.*[\r\n]*', '', text)

def remove_symbols(text):
    return re.sub(r'([^A-Za-z0-9\'])', ' ', text)

def convert_entities(text):
    parser = HTMLParser.HTMLParser()
    return parser.unescape(text)

def convert_case(text):
    return text.lower()


# Open Training Files
DevelopBankFile =  open("DBS1.txt", "rb")
DefensiveBacksFile =  open("DBS2.txt", "rb")
NationalUniversityFile = open("NUS1.txt", "rb")
NationalUnionFile = open("NUS2.txt", "rb")
StarhubFile = open("STARHUB.txt", "rb")

# Read all Training Files
DevelopBankJSON = [i for i in DevelopBankFile.readlines()]
DefensiveBacksJSON = [i for i in DefensiveBacksFile.readlines()]
NationalUniversityJSON = [i for i in NationalUniversityFile.readlines()]
NationalUnionJSON = [i for i in NationalUnionFile.readlines()]
StarhubJSON = [i for i in StarhubFile.readlines()]


DevelopBankSet = set()
DefensiveBacksSet = set()
NationalUniversitySet = set()
NationalUnionSet = set()
StarhubSet = set()

# Load all n-gram into set
for i in DevelopBankJSON:
    tmp = json.loads(i,  encoding="utf-8")
    txt = tmp["text"].encode('ascii','ignore')
    txt = remove_mentions(txt)
    txt = remove_links(txt)
    txt = remove_symbols(txt)
    txt = convert_entities(txt)
    txt = convert_case(txt)
    DevelopBankSet.update(txt.split())

for i in DefensiveBacksJSON:
    tmp = json.loads(i,  encoding="utf-8")
    txt = tmp["text"].encode('ascii','ignore')
    txt = remove_mentions(txt)
    txt = remove_links(txt)
    txt = remove_symbols(txt)
    txt = convert_entities(txt)
    txt = convert_case(txt)
    DefensiveBacksSet.update(txt.split())

for i in NationalUniversityJSON:
    tmp = json.loads(i,  encoding="utf-8")
    txt = tmp["text"].encode('ascii','ignore')
    txt = remove_mentions(txt)
    txt = remove_links(txt)
    txt = remove_symbols(txt)
    txt = convert_entities(txt)
    txt = convert_case(txt)
    NationalUniversitySet.update(txt.split())

for i in NationalUnionJSON:
    tmp = json.loads(i,  encoding="utf-8")
    txt = tmp["text"].encode('ascii','ignore')
    txt = remove_mentions(txt)
    txt = remove_links(txt)
    txt = remove_symbols(txt)
    txt = convert_entities(txt)
    txt = convert_case(txt)
    NationalUnionSet.update(txt.split())

for i in StarhubJSON:
    tmp = json.loads(i,  encoding="utf-8")
    txt = tmp["text"].encode('ascii','ignore')
    txt = remove_mentions(txt)
    txt = remove_links(txt)
    txt = remove_symbols(txt)
    txt = convert_entities(txt)
    txt = convert_case(txt)
    StarhubSet.update(txt.split())

print len(DevelopBankSet)
print len(DefensiveBacksSet)
print len(NationalUniversitySet)
print len(NationalUnionSet)
print len(StarhubSet)
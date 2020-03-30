#APIKey = AIzaSyAQiB8uDcVFQ4HEXZtGfJ1YozGq3CqLWZ0
#searchID = 018055869393845110026:w2dl6jqrm17
from googleapiclient.discovery import build
from collections import defaultdict
import sys
import numpy as np
from itertools import permutations
import argparse
from stanfordnlp.server import CoreNLPClient
from stanfordnlp.server import to_text
import tika
from tika import parser
from decimal import *

fakeX = {
    ('Mary Maxwell Gates', 'per:employee_or_member_of', 'Cascade Investment Technology'): 0.9484856396848951, 
    ('Melinda French', 'per:employee_or_member_of', 'Bill & Melinda Gates Foundation'): 0.8882812681198851, 
    ('William Henry Gates III', 'per:employee_or_member_of', 'Branded Entertainment Network'): 0.910425809655642, 
    ('Mary Maxwell Gates', 'per:employee_or_member_of', 'Microsoft'): 0.9535132816229053, 
    ('William Henry Gates III', 'per:employee_or_member_of', 'Microsoft'): 0.9605363777239155, 
    ('Bill Gates Gates', 'per:employee_or_memberof', 'Cascade Investment Technology'): 0.7307587220750844, 
    ('Melinda French', 'per:employeeor_member_of', 'Branded Entertainment Network'): 0.8205988757633852, 
    ('Bill Gates Gates', 'per:employee_or_member_of', 'Branded Entertainment Network'): 0.8528370731813597, 
    ('William Henry Gates III', 'per:employee_or_member_of', 'Bill & Melinda Gates Foundation'): 0.964341523275631, 
    ('Mary Maxwell Gates', 'per:employee_or_member_of', 'Bill & Melinda Gates Foundation'): 0.9596297625973506, 
    ('Melinda French', 'per:employee_or_member_of', 'Cascade Investment Technology'): 0.9331336130042003, 
    ('Bill Gates Gates', 'per:employee_or_member_of', 'Microsoft'): 0.9789945486751996, 
    ('Bill Gates Gates', 'per:employee_or_member_of', 'Bill & Melinda Gates Foundation'): 0.9592873304873829, 
    ('Melinda French', 'per:employee_or_member_of', 'Microsoft'): 0.9561346424552023, 
    ('Mary Maxwell Gates', 'per:employee_or_member_of', 'Branded Entertainment Network'): 0.8958521246315242, 
    ('Paul Allen', 'per:employee_or_member_of', 'Microsoft'): 0.7123580527943317, 
    ('Gates', 'per:employee_or_member_of', 'Microsoft'): 1.0, 
    ('He', 'per:employee_or_member_of', 'Microsoft'): 1.0, 
    ('Gates', 'per:employee_or_member_of', 'Harvard'): 1.0, 
    ('Ballmer', 'per:employee_or_member_of', 'Microsoft'): 0.6911532514802464, 
    ('Christos Papadimitriou', 'per:employee_or_member_of', 'Harvard'): 1.0, 
    ('Allen', 'per:employee_or_member_of', 'Intel'): 0.6400002269572793, 
    ('Gates', 'per:employee_or_member_of', 'Intel'): 0.7639629571437762, 
    ('Jack Sams', 'per:employee_or_member_of', 'IBM'): 1.0, 
    ('Tim Paterson', 'per:employee_or_member_of', 'Seattle Computer Products'): 1.0, 
    ('Post-Microsoft Gates', 'per:employee_or_member_of', 'Microsoft'): 0.7175707843712362, 
    ('Satya Nad', 'per:employee_or_member_of', 'Microsoft'): 0.6728623638353038, 
    ('he', 'per:employee_or_member_of', 'Cascade Investment LLC'): 0.6196949364043983, 
    ('Gates', 'per:employee_or_member_of', 'Corbis'): 0.6027934517059498
}


### HI GARY
"""

THINGS WE STILL NEED:

- last 2 annotations (wasn't sure where to put them) which are below:

print("\tExtracted kbp annotations for 17 out of total 39 sentences") <- swap with real variables
print("\bRelations extracted from this website: 5 (Overall: 5)") <- swap with real variables

"""


def main(APIkey, engineID, r, t, Q, k):

    # ignore but don't remove
    relationdict = {'1':"per:schools_attended",'2':"per:employee_or_member_of",'3':"per:cities_of_residence",'4':"org:top_members_employees"}

    # Initial print statements
    print("\n\n____")
    print("Parameters:")
    print("Client key\t=", APIkey)
    print("Engine key\t=", engineID)
    print("Relation\t=", relationdict[r])
    print("Threshold\t=", t)
    print("Query\t\t=", Q)
    print("# of Tuples\t=", k)
    print("Loading necessary libraries; This should take a minute or so ...)")



    X = defaultdict(float) #key is tuple, value is confidence value, this will help with duplicates and update
    prev_query_words = set()
    iter_count = 0

    while len(X) < int(k):
        #update the list of past query words
        for word in Q.split(" "):
            prev_query_words.add(word.lower())

        ### remove the below comment and disregard fake X (for testing only)
        #X = step3(APIkey, engineID, r, t, Q, k)
        print("=========== Iteration:",iter_count,"- Query:",Q,"===========")
        X = fakeX

        # sort X in decreasing intervals of extraction confidence
        # sorted_X is a list not dict
        sorted_X = sorted(X.items(), key=lambda item: item[1], reverse=True)

        # step 6: select new tuple to add to query, based on top extraction confidence level
        y = tuple()
        for tup, val in sorted_X:
            y = tup
            sub = tup[0]
            obj = tup[2]

            # is subject in query? if so, skip
            if sub.lower() in prev_query_words:
                continue
            # is object in the query? if so, skip
            elif obj.lower() in prev_query_words:
                continue
            # if neither are in query, then choose this y
            else:
                break
        # now y tuple is set, in case we make another query search

        # if no possible y then just stop
        if len(y) == 0:
            break
        new_query = str(Q) + " " + y[0] + " " + y[2]
        Q = new_query
        iter_count += 1

    # The above loop ends when X contains at least k tuples

    if len(X) >= int(k):
        # step 5: return the tuples with their extraction confidence
        print("================== ALL RELATIONS (",len(X),") =================")
        getcontext().prec = 6
        for key, value in sorted_X:
            print("Confidence:",Decimal(str(value)),"\t\t| Subject:", key[0],"\t\t| Object:",key[2])
    else:
        print("No results found.")
    

def step3(APIkey, engineID, r, t, Q, k):
    # now begin searching query
    relationdict = {'1':"per:schools_attended",'2':"per:employee_or_member_of",'3':"per:cities_of_residence",'4':"org:top_members_employees"}
    service = build("customsearch", "v1", developerKey=APIkey)
    res = service.cse().list(q=Q, cx=engineID, num=5).execute()
    X = process_urls(res,relationdict,r,t)
    return(X)

def process_urls(res,relationdict,r,t):
    name_dict = {'1':["ORGANIZATION","PERSON"],'2':["ORGANIZATION","PERSON"],'3':["LOCATION","CITY","STATE_OR_PROVINCE","COUNTRY"],'4':["ORGANIZATION","PERSON"]}
    X = defaultdict(float)
    count = 1
    for results in res['items']:
        url = results['link']
        title = results['title']
        print("URL (",count,"/ 10):", url)
        print("\tFetching text from url ...")

        ## gary please fill in
        webpage_length = 0

        #print('result ' + str(count))
        #print('title: '+title)
        #print('url: '+url)
        count += 1
        #now get contents using tika
        try:
            parsed = parser.from_file(url)
        except:
            continue
        content = parsed["content"]
        strip_content = ' '.join(content.split())
        print(len(strip_content))
        #get first 20000 characters
        if len(strip_content) >= 20000:
            strip_content = strip_content[:20000]
        print("\tWebpage length (num characters):", len(strip_content))
        print("\tAnnotating the webpage using [tokenize, ssplit, pos, lemma, ner] annotators ...")
        #use package now

        with CoreNLPClient(annotators=['tokenize', 'ssplit', 'pos', 'lemma', 'ner'], timeout=30000, memory='4G',
                           endpoint="http://localhost:9000") as pipeline_ner:
            with CoreNLPClient(annotators=['tokenize', 'ssplit', 'pos', 'lemma', 'ner', 'depparse', 'coref', 'kbp'],
                               timeout=30000, memory='4G', endpoint="http://localhost:9001") as pipeline_kbp:
                for j in range(10): # this number can be changed back to 10 if needed
                    try:
                        #print(f">>> Repeating {j}th time.")
                        ann_ner = pipeline_ner.annotate(strip_content)
                        #print("getting to annotate")

                        print("\tExtracted", len(ann_ner.sentence), "sentences. Processing each sentence one by one to check for presence of right pair of named entity types; if so, will run the second pipeline ...")
                        for sentence in ann_ner.sentence:
                            #print("matching ners...")
                            match = [False] * len(name_dict[r])
                            for token in sentence.token:
                                for i in range(len(name_dict[r])):
                                    if token.ner == name_dict[r][i]:
                                        match[i] = True
                            if all(match):
<<<<<<< HEAD
                                print("match success, begin kbp")

=======
                                #print("match success, begin kbp")
>>>>>>> 3ec98e4574e064ce0f352813ae0a3e850534f9d9
                                ann = pipeline_kbp.annotate(to_text(sentence))
                                countsentences = 0
                                for i in ann.sentence:
                                    if (countsentences % 5 == 0) and (countsentences != 0):
                                        print("\tProcessed", countsentences+1,"/", len(ann_ner.sentence),"sentences")
                                    elif (countsentences == len(ann.sentence)):
                                        print("\tProcessed", len(ann_ner.sentence),"/", len(ann_ner.sentence),"sentences")
                                    for kbp_triple in i.kbpTriple:
                                        if kbp_triple.relation == relationdict[r]:
                                            #print("relation match")
                                            print("\t\t=== Extracted Relation ===")
                                            print("\t\tSentence:", i) ########## not sure if this is right, please check
                                            print("\t\tConfidence:", kbp_triple.confidence,"; Subject:", kbp_triple.subject, "; Object:", kbp_triple.object, ";")
                                            if kbp_triple.confidence > float(t):
                                                if X[(kbp_triple.subject,kbp_triple.relation,kbp_triple.object)] < kbp_triple.confidence:
                                                    #update key value now
                                                    X[(kbp_triple.subject,kbp_triple.relation,kbp_triple.object)] = kbp_triple.confidence
                                                    print("\t\tAdding to set of extracted relations")
                                                    #print((kbp_triple.subject,kbp_triple.relation,kbp_triple.object,kbp_triple.confidence))
                                            else:
                                                print("\t\tConfidence is lower than threshold confidence. Ignoring this.")
                                            print("\t\t==========")
                                    countsentences += 1
                        break
                    except:
                        continue
                print(X.items()) #test
    return X





if __name__ == '__main__':
    APIkey = sys.argv[1]
    engineID = sys.argv[2]
    r = sys.argv[3]
    q = sys.argv[5]
    t = sys.argv[4]
    k = sys.argv[6]
    main(APIkey, engineID, r, t, q, k)
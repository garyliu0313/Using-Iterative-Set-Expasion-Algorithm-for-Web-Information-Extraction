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
    sorted_X = []
    prev_query_words = set()
    iter_count = 0

    while len(X) < int(k):
        #update the list of past query words
        for word in Q.split(" "):
            prev_query_words.add(word.lower())

        ### remove the below comment and disregard fake X (for testing only)
        print("=========== Iteration:",iter_count,"- Query:",Q,"===========")

        X = step3(APIkey, engineID, r, t, Q, k)
        # sort X in decreasing intervals of extraction confidence
        # sorted_X is a list not dict
        sorted_X = sorted(X.items(), key=lambda item: item[1], reverse=True)
        # step 5: return the tuples with their extraction confidence
        print("================== ALL RELATIONS (", len(X), ") =================")
        getcontext().prec = 6
        for key, value in sorted_X:
            print("Confidence:", Decimal(str(value)), "\t\t| Subject:", key[0], "\t\t| Object:", key[2])
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
        new_query = y[0] + " " + y[2]
        Q = new_query
        iter_count += 1

    # The above loop ends when X contains at least k tuples

    if len(X) < int(k):
        print("No results found.")
    

def step3(APIkey, engineID, r, t, Q, k):
    # now begin searching query
    relationdict = {'1':"per:schools_attended",'2':"per:employee_or_member_of",'3':"per:cities_of_residence",'4':"org:top_members_employees"}
    service = build("customsearch", "v1", developerKey=APIkey)
    res = service.cse().list(q=Q, cx=engineID, num=10).execute()
    X = process_urls(res,relationdict,r,t)
    return(X)

def process_urls(res,relationdict,r,t):
    name_dict = {'1':["ORGANIZATION","PERSON"],'2':["ORGANIZATION","PERSON"],'3':["PERSON","CITY"],'4':["ORGANIZATION","PERSON"]}
    X = defaultdict(float)
    count = 1
    with CoreNLPClient(annotators=['tokenize', 'ssplit', 'pos', 'lemma', 'ner'], timeout=30000, memory='4G',
                       endpoint="http://localhost:9000") as pipeline_ner:
        with CoreNLPClient(annotators=['tokenize', 'ssplit', 'pos', 'lemma', 'ner', 'depparse', 'coref', 'kbp'],
                           timeout=30000, memory='4G', endpoint="http://localhost:9001") as pipeline_kbp:
            for results in res['items']:
                url = results['link']
                title = results['title']
                print("URL (", count, "/ 10):", url)
                print("\tFetching text from url ...")

                ## gary please fill in
                webpage_length = 0

                # print('result ' + str(count))
                # print('title: '+title)
                # print('url: '+url)
                count += 1
                # now get contents using tika
                try:
                    parsed = parser.from_file(url)
                except:
                    print("Unable to fetch URL. Continuing.")
                    continue
                content = parsed["content"]
                strip_content = ' '.join(content.split())
                print(len(strip_content))
                # get first 20000 characters
                if len(strip_content) >= 20000:
                    strip_content = strip_content[:20000]
                print("\tWebpage length (num characters):", len(strip_content))
                print("\tAnnotating the webpage using [tokenize, ssplit, pos, lemma, ner] annotators ...")
                # use package now

                # for j in range(10): # this number can be changed back to 10 if needed
            #     try:
                    #print(f">>> Repeating {j}th time.")
                ann_ner = pipeline_ner.annotate(strip_content)
                countkbp = 0
                print("\tExtracted", len(ann_ner.sentence), "sentences. Processing each sentence one by one to check for presence of right pair of named entity types; if so, will run the second pipeline ...")
                for sentence in ann_ner.sentence:
                    #print("matching ners...")
                    match = [False] * len(name_dict[r])
                    for token in sentence.token:
                        for i in range(len(name_dict[r])):
                            if token.ner == name_dict[r][i]:
                                match[i] = True
                    if all(match):
                        try:
                            ann = pipeline_kbp.annotate(to_text(sentence))
                        except:
                            continue
                        countsentences = 0
                        countkbp += 1
                        for i in ann.sentence:
                            if (countsentences % 5 == 0) and (countsentences != 0):
                                print("\tProcessed", countsentences+1,"/", len(ann_ner.sentence),"sentences")
                            elif (countsentences == len(ann.sentence)):
                                print("\tProcessed", len(ann_ner.sentence),"/", len(ann_ner.sentence),"sentences")
                            for kbp_triple in i.kbpTriple:
                                if kbp_triple.relation == relationdict[r]:

                                    print("\t\t=== Extracted Relation ===")
                                    print("\t\tSentence:", to_text(i)) ########## not sure if this is right, please check
                                    print("\t\tConfidence:", kbp_triple.confidence,"; Subject:", kbp_triple.subject, "; Object:", kbp_triple.object, ";")
                                    if kbp_triple.confidence > float(t):
                                        if X[(kbp_triple.subject,kbp_triple.relation,kbp_triple.object)] < kbp_triple.confidence:
                                            #update key value now
                                            X[(kbp_triple.subject,kbp_triple.relation,kbp_triple.object)] = kbp_triple.confidence
                                            print("\t\tAdding to set of extracted relations")
                                            #print((kbp_triple.subject,kbp_triple.relation,kbp_triple.object,kbp_triple.confidence))
                                        else:
                                            print("\t\tDuplicate with lower confidence than existing record. Ignoring this.")
                                    else:
                                        print("\t\tConfidence is lower than threshold confidence. Ignoring this.")
                                    print("\t\t==========")
                            countsentences += 1
                print("\tExtracted kbp annotations for ",countkbp," out of total ",len(ann_ner.sentence)," sentences")

    return X





if __name__ == '__main__':
    APIkey = sys.argv[1]
    engineID = sys.argv[2]
    r = sys.argv[3]
    q = sys.argv[5]
    t = sys.argv[4]
    k = sys.argv[6]
    main(APIkey, engineID, r, t, q, k)
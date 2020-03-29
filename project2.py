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


def main(APIkey, engineID, r, t, Q, k):
    X = defaultdict(float) #key is tuple, value is confidence value, this will help with duplicates and update
    relationdict = {'1':"per:schools_attended",'2':"per:employee_or_member_of",'3':"per:cities_of_residence",'4':"org:top_members_employees"}

    # now begin searching query
    service = build("customsearch", "v1", developerKey=APIkey)
    res = service.cse().list(q=Q, cx=engineID, num=5).execute()
    X = process_urls(res,relationdict,r,t)
    print(X)
    print("done first 3 part")


def process_urls(res,relationdict,r,t):
    name_dict = {'1':["ORGANIZATION","PERSON"],'2':["ORGANIZATION","PERSON"],'3':["LOCATION","CITY","STATE_OR_PROVINCE","COUNTRY"],'4':["ORGANIZATION","PERSON"]}
    X = defaultdict(float)
    count = 1
    for results in res['items']:
        url = results['link']
        title = results['title']
        print('----------------------------------------')
        print('result ' + str(count))
        print('title: '+title)
        print('url: '+url)
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
        print(len(strip_content))
        #use package now

        with CoreNLPClient(annotators=['tokenize', 'ssplit', 'pos', 'lemma', 'ner'], timeout=30000, memory='4G',
                           endpoint="http://localhost:9000") as pipeline_ner:
            with CoreNLPClient(annotators=['tokenize', 'ssplit', 'pos', 'lemma', 'ner', 'depparse', 'coref', 'kbp'],
                               timeout=30000, memory='4G', endpoint="http://localhost:9001") as pipeline_kbp:
                for j in range(3): # this number can be changed back to 10 if needed
                    print(f">>> Repeating {j}th time.")
                    ann_ner = pipeline_ner.annotate(strip_content)
                    print("getting to annotate")
                    for sentence in ann_ner.sentence:
                        print("matching ners...")
                        match = [False] * len(name_dict[r])
                        for token in sentence.token:
                            for i in range(len(name_dict[r])):
                                if token.ner == name_dict[r][i]:
                                    match[i] = True
                        if all(match):
                            print("match success, begin kbp")
                            ann = pipeline_kbp.annotate(to_text(sentence))
                            for i in ann.sentence:
                                for kbp_triple in i.kbpTriple:
                                    if kbp_triple.relation == relationdict[r]:
                                        print("relation match")
                                        if kbp_triple.confidence > float(t):
                                            if X[(kbp_triple.subject,kbp_triple.relation,kbp_triple.object)] < kbp_triple.confidence:
                                                #update key value now
                                                X[(kbp_triple.subject,kbp_triple.relation,kbp_triple.object)] = kbp_triple.confidence
                                                print("inserted:")
                                                print((kbp_triple.subject,kbp_triple.relation,kbp_triple.object,kbp_triple.confidence))
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
#APIKey = AIzaSyAQiB8uDcVFQ4HEXZtGfJ1YozGq3CqLWZ0
#searchID = 018055869393845110026:w2dl6jqrm17
from googleapiclient.discovery import build
from collections import defaultdict
import sys
import numpy as np
from itertools import permutations
import argparse
from stanfordnlp.server import CoreNLPClient
import tika
from tika import parser


def main(APIkey, engineID, r, t, Q, k):
    X = defaultdict(float) #key is tuple, value is confidence value, this will help with duplicates and update
    relationdict = {1:"per_schools_attended",2:"per_employee_of",3:"per_cities_of_residence",4:"org_top_members/employees"}

    # now begin searching query
    service = build("customsearch", "v1", developerKey=APIkey)
    res = service.cse().list(q=Q, cx=engineID, num=5).execute()
    X = process_urls(res,relationdict)
    print(X)


def process_urls(res,relationdict):
    X = defaultdict(float)
    count = 1
    for results in res['items']:
        url = results['link']
        title = results['title']
        print('----------------------------------------')
        print('result ' + str(count))
        print('title: '+title)
        print('url: '+url)
        #now get contents using tika
        try:
            parsed = parser.from_file(url)
        except:
            continue
        content = parsed["content"]
        strip_content = ' '.join(content.split())
        print(type(strip_content))
        print(strip_content)
        #get first 20000 characters
        if len(strip_content) >= 20000:
            strip_content = strip_content[:20000]
        #use package now

        # argparser = argparse.ArgumentParser()
        # argparser.add_argument("--input_text_file_path", "-i", required=True, help="Path to the text file to be parsed")
        # args = argparser.parse_args()
        #
        # with open(args.input_text_file_path, "r") as f:
        #     text = f.read()

        with CoreNLPClient(annotators=['tokenize', 'ssplit', 'pos', 'lemma', 'ner'], timeout=30000, memory='4G',
                           endpoint="http://localhost:9000") as pipeline_ner:
            with CoreNLPClient(annotators=['tokenize', 'ssplit', 'pos', 'lemma', 'ner', 'depparse', 'coref', 'kbp'],
                               timeout=30000, memory='4G', endpoint="http://localhost:9001") as pipeline_kbp:
                for j in range(100):
                    print(f">>> Repeating {j}th time.")
                    ann_ner = pipeline_ner.annotate(strip_content)
                    ann = pipeline_kbp.annotate(strip_content)
                    for sentence in ann.sentence:
                        for kbp_triple in sentence.kbpTriple:
                            print(
                                f"\t Confidence: {kbp_triple.confidence};\t "
                                f"Subject: {kbp_triple.subject};\t Relation: {kbp_triple.relation}; "
                                f"Object: {kbp_triple.object}") #test
                            # extract relation
                            if kbp_triple.relation == relationdict[r]:
                                print("relation match") #test
                                if X[(kbp_triple.subject,kbp_triple.relation,kbp_triple.object)] < kbp_triple.confidence\
                                        and kbp_triple.confidence > t:#update key value now
                                    X[(kbp_triple.subject,kbp_triple.relation,kbp_triple.object)] = kbp_triple.confidence
                print(X.items()) #test
    return X
if __name__ == '__main__':
    APIkey = sys.argv[1]
    engineID = sys.argv[2]
    r = sys.argv[3]
    t = sys.argv[4]
    q = sys.argv[5]
    k = sys.argv[6]
    main(APIkey, engineID, r, t, q, k)
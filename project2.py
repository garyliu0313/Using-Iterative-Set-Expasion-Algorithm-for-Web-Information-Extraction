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


def main(APIkey, engineID, r, t, q, k):
    X = defaultdict(dict) #key is tuple, value is confidence value, this will help with duplicates and update

    service = build("customsearch", "v1", developerKey=APIkey)
    res = service.cse().list(q=Q, cx=searchID, num=10).execute()

    for results in res['items']:
        url = results['link']
        title = results['title']
        summary = results['snippet']
        print('----------------------------------------')
        print('result ' + str(count))
        print('title: '+title)
        print('url: '+url)
        print('summary: '+summary)
        #now get contents using beautiful soap
        parsed = parser.from_file(url)
        content = parsed["content"]
        strip_content = ' '.join(content.split())
        #get first 20000 characters
        if len(strip_content) >= 20000:
            strip_content = strip_content[:20000]
        #use package now,only copied the example for now.
        argparser = argparse.ArgumentParser()
        argparser.add_argument("--input_text_file_path", "-i", required=True, help="Path to the text file to be parsed")
        args = argparser.parse_args()

        with open(args.input_text_file_path, "r") as f:
            text = f.read()

        with CoreNLPClient(annotators=['tokenize', 'ssplit', 'pos', 'lemma', 'ner'], timeout=30000, memory='4G',
                           endpoint="http://localhost:9000") as pipeline_ner:
            with CoreNLPClient(annotators=['tokenize', 'ssplit', 'pos', 'lemma', 'ner', 'depparse', 'coref', 'kbp'],
                               timeout=30000, memory='4G', endpoint="http://localhost:9001") as pipeline_kbp:
                for j in range(100):
                    print(f">>> Repeating {j}th time.")
                    ann_ner = pipeline_ner.annotate(text)
                    ann = pipeline_kbp.annotate(text)
                    for sentence in ann.sentence:
                        for kbp_triple in sentence.kbpTriple:
                            print(
                                f"\t Confidence: {kbp_triple.confidence};\t "
                                f"Subject: {kbp_triple.subject};\t Relation: {kbp_triple.relation}; "
                                f"Object: {kbp_triple.object}")


if __name__ == '__main__':
    APIkey = sys.argv[1]
    searchID = sys.argv[2]
    r = sys.argv[3]
    t = sys.argv[4]
    q = sys.argv[5]
    k = sys.argv[6]
    main(APIkey, engineID, r, t, q, k)
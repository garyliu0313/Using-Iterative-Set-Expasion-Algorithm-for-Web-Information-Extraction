#APIKey = AIzaSyAQiB8uDcVFQ4HEXZtGfJ1YozGq3CqLWZ0
#searchID = 018055869393845110026:w2dl6jqrm17
from googleapiclient.discovery import build
from collections import defaultdict
import sys
import numpy as np
from itertools import permutations
from stanfordnlp.server import CoreNLPClient

def main(APIkey, engineID, r, t, q, k):
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

        #get first 20000 characters

        #use package now



if __name__ == '__main__':
    APIkey = sys.argv[1]
    searchID = sys.argv[2]
    r = sys.argv[3]
    t = sys.argv[4]
    q = sys.argv[5]
    k = sys.argv[6]
    main(APIkey, engineID, r, t, q, k)
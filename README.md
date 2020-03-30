# cs6111-project2
===========
 Samantha MacIlwaine srm2197
 Gary Liu zl2889
===========

=== List of all files submitting ===
project2.py
README.md
transcript.txt

=== Description of how to run program ===
# To get package that we are using, run following commands:
sudo apt-get update

# Get Stanford CoreNLP
wget http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip
sudo apt-get install unzip
unzip stanford-corenlp-full-2018-10-05.zip

# Install Java 13
wget https://download.java.net/java/GA/jdk13.0.2/d4173c853231432d94f001e99d882ca7/8/GPL/openjdk-13.0.2_linux-x64_bin.tar.gz
tar -xvzf openjdk-13.0.2_linux-x64_bin.tar.gz

# Set PATH and JAVA_HOME
export PATH=/home/⟨your_UNI⟩/jdk-13.0.2/bin:$PATH
export JAVA_HOME=/home/⟨your_UNI⟩/jdk-13.0.2

# Install packages
sudo apt-get install python3-pip
pip3 install google-api-python-client
pip3 install scikit-learn
pip3 install tika
pip3 install stanfordnlp

# Set CORENLP_HOME
export CORENLP_HOME=/home/⟨your_UNI⟩/stanford-corenlp-full-2018-10-05

# Command to run
python3 project2.py <google api key> <google engine id> <r> <t> <q> <k>

=== Description of Internal methods ===

The input is received through the last-written function at the bottom
of the page. Then main() is called.

*** STEP 1 ***
In our program, X is a dictionary in which the tuples are the keys
and the extraction confidence is the value. The main() function initializes
X as the empty dictionary.

Then, main() initiates a while loop that handles steps 2-6.

*** STEP 2 ***
In the while loop, until X contains at least k tuples (which will run
at least once, since X starts out containing zero tuples), the step3() 
method is called with the query q and other arguments.

In step3(), the Google Custom Search Engine is run, and the results are
passed to process_urls().

*** STEP 3 ***
For each URL in the response object res[], we extracted the content using tika
and feed them to the ner annotator first. Then we maintain a matching list to
determine whether a sentence contains all the ner tag we need for extracting
the relation. Then we use kbp annotator to annotate the matched sentences and
added the tuples that satisfy given confidence level and relation to the
dictionary X.

*** STEP 4 ***
Step 4 is performed within the process_urls() loop, where kbp_triple.confidence,
for a particular tuple, is only added to X if it is greater than the existing
value for that tuple (which is the key). This prevents duplicate entries
automatically.

*** STEP 5 and 6 ***
The while loop in Step 2 breaks when len(X) is at least as great as k, meaning
that X contains at least k tuples. 

For each iteration that it does not break, a new query is prepared by sorting X
and finding the tuple y that has the highest extraction confidence value and
whose subject and object have not been seen in any previous queries.

The sorted tuples are printed in decreasing order of extraction confidence on
each iteration of the loop.

=== Keys ===
Google Custom Search Engine JSON API Key = AIzaSyAQiB8uDcVFQ4HEXZtGfJ1YozGq3CqLWZ0
Engine ID = 018055869393845110026:w2dl6jqrm17







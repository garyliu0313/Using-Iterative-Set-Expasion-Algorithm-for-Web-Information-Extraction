# cs6111-project2
===========
 Samantha MacIlwaine srm2197
 Gary Liu zl2889
===========

=== List of all files submitting ===
project2.py
README.md

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

=== Description of Internal methods ===

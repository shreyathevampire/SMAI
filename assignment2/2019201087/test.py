from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

from q5 import AuthorClassifier as ac
auth_classifier = ac()
auth_classifier.train('./Datasets/q5/train.csv') # Path to the train.csv will be provided
predictions = auth_classifier.predict('./Datasets/q5/test.csv') # Path to the test.csv will be provided

'''WE WILL CHECK THE PREDICTIONS WITH THE GROUND TRUTH LABELS'''

from q6 import Cluster as cl
cluster_algo = cl()
# You will be given path to a directory which has a list of documents. You need to return a list of cluster labels for those documents
predictions = cluster_algo.cluster('./Datasets/q6/') 

'''SCORE BASED ON THE ACCURACY OF CLUSTER LABELS'''

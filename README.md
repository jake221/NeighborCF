
# Description

## kNNItemCF.py
item-based collaborative filtering algorithm for implicit feedback data, evaluated by precision, recall, ndcg and map.
## kNNUserCF.py
user-based collaborative filtering algorithm for implicit feedback data, evaluated by precision, recall, ndcg and map.
## kNNItemCF.py
item-based collaborative filtering algorithm for implicit feedback data, evaluated by AUC.
## map.py
map measure
## ndcg.py
ndcg measure

# Data
## ua.base
training set
## ua.test
test set. 
They are downloaded from GroupLens official website, http://files.grouplens.org/datasets/movielens/.

# Metrics
kNNItemCF and kNNUserCF are evaluated by precision, recall, nDCG and map.
kNNItemCF_AUC is evaluated by AUC.


# Requirement
You need to download numpy to run this program.

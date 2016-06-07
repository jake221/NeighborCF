# coding : UTF-8
__author__ = 'wangjinkun@mail.hfut.edu.cn'

import math
import numpy as np

# ndcg for plancast dataset: where only one item exists in each group
# as only one item exists in each group, so the idcg is always 1.
# and the dcg is equal to the rank of the right item's discounted cumulative gain

rec_list = [[7,10,21,13,30,12,16,29,30,35],[8,20,21,32,43,12,16,29,30,35]]
test_list = [[7],[32]]
# number of users
ndcg = 0.0
for i in range(len(rec_list)):
    if len(np.intersect1d(rec_list[i],test_list[i])) != 0:
        rec_rank = rec_list[i].index(test_list[i][0])
        ndcg = ndcg + (1.0 / math.log(rec_rank+2,2))
        print rec_rank,ndcg
ndcg = ndcg / len(rec_list)
print ndcg

# ndcg for movielens dataset: where each user possess ten items in test data.
# dcg = \sum_i=1^N \frac{2^rel_i - 1}{log(i,4)}
# rec_list = [[7,10,21,13,30,12,16,29,30,35],[8,20,21,32,43,12,16,29,30,35]]
# test_list = [[35,41,30,20,21,36,28,7,3,1],[5,41,30,20,21,36,28,7,3,1]]
# rec_list = [[7,10,21,13,30,12,16,29,30,35]]
# test_list = [[35,41,30,20,21,36,28,7,3,1]]
rec_list = [[7,10,21,13,30,12,16,29,30,35],[8,20,21,32,43,12,16,29,30,35]]
test_list = [[7],[32]]
# number of users
num_users = len(rec_list)
idcg = np.zeros(num_users)
dcg = np.zeros(num_users)
ndcg = np.zeros(num_users)
for i in range(num_users):
    a = np.intersect1d(rec_list[i],test_list[i])    # the hit item set
    if len(a) != 0:
        rel_list = np.zeros((len(rec_list[0])))
        rank = 0.0    # to calculate idcg
        for item in a:
            item_rank = rec_list[i].index(item)     # relevant items in the rec_list, to calculate dcg
            rel_list[item_rank] = 1
            dcg[i] = dcg[i] + 1 / math.log(item_rank+2,2)
            idcg[i] = idcg[i] + 1 / math.log(rank+2,2)
            rank = rank + 1
            print item_rank,dcg,idcg
    ndcg[i] = dcg[i] / (idcg[i] * 1.0)
    print ndcg
ndcg = sum(ndcg) / len(rec_list)
print ndcg

# coding : UTF-8
__author__ = 'wangjinkun@mail.hfut.edu.cn'

import numpy as np

# map for movielens dataset: where each user possess ten items in test data.
rec_list = [[7,10,21,13,30,12,16,29,30,35],[8,20,21,32,43,12,16,29,30,35]]
test_list = [[35,41,30,20,21,36,28,7,3,1],[5,41,30,20,21,36,28,7,3,1]]
# rec_list = [[7,10,21,13,30,12,16,29,30,35]]
# test_list = [[35,41,30,20,21,36,28,7,3,1]]
# rec_list = [[7,10,21,13,30,12,16,29,30,35],[8,20,21,32,43,12,16,29,30,35]]
# test_list = [[7],[32]]
# number of users
num_users = len(rec_list)
map = np.zeros(num_users)
for i in range(num_users):
    a = np.intersect1d(rec_list[i],test_list[i])    # the hit item set
    print a
    if len(a) != 0:
        rel_list = np.zeros((len(rec_list[0])))
        rank = 1.0    # to calculate idcg
        for item in a:
            item_rank = rec_list[i].index(item)     # relevant items in the rec_list, to calculate dcg
            map[i] = map[i] + rank / (item_rank + 1.0)
            rank = rank + 1.0

            print item,item_rank,map
    map[i] = map[i] / len(a)
    print map
print sum(map) / num_users

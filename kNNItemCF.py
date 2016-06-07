# coding = utf-8
# Please feel free to contact with me if you have any question with the code.
__author__ = 'wangjinkun@mail.hfut.edu.cn'

import numpy as np
import time

def load_matrix(filename, num_users, num_items):
    t0 = time.time()
    matrix = np.zeros((num_users,num_items))
    for line in open(filename):
        user,item,_,_ = line.split()
        user = int(user)
        item = int(item)
        count = 1.0
        matrix[user-1,item-1] = count
    t1 = time.time()
    print 'Finished loading matrix in %f seconds' % (t1-t0)
    return  matrix

class ItemCF:

    def __init__(self,traindata,testdata):
        self.traindata = traindata
        self.testdata = testdata
        self.num_users = traindata.shape[0]
        self.num_items = traindata.shape[1]

    def ItemSimilarity(self):
        t0 = time.time()
        train = self.traindata
        num_items = self.num_items
        self.item_similarity = np.zeros((num_items,num_items))

        for i in np.arange(0,num_items):
            r_i = train[:,i]
            self.item_similarity[i,i] = 0
            for j in np.arange(i+1,num_items):
                r_j = train[:,j]
                num = np.dot(r_i.T , r_j)
                denom = np.linalg.norm(r_i) * np.linalg.norm(r_j)
                if denom == 0:
                    cos = 0
                else:
                    cos = num / denom
                self.item_similarity[i,j] = cos
                self.item_similarity[j,i] = cos
        self.item_neighbor = np.argsort(-self.item_similarity)
        t1 = time.time()
        print 'Finished calculating similarity matrix in %f seconds' % (t1-t0)

    def Recommendation(self,user_id,kNN,top_N):
        # recommend a top_N recommendation list for user_id
        # r_ui = \sum_{j \in N^k(i)} r_uj \times w_ij
        train = self.traindata
        similarity = self.item_similarity

        # find the user's rating history
        r_u = train[user_id]
        rated_items = np.nonzero(r_u)
        rated_items_idx = rated_items[0]    # items rated by user_id in train set
        predict_items_idx = np.setdiff1d(np.arange(0,self.num_items),rated_items_idx)   # item index that has to be predicted
        pred_score = np.zeros((1,self.num_items))
        for i in predict_items_idx:
            item_idx = i
            neighbor_ordered = self.item_neighbor[item_idx] #
            for neigh in neighbor_ordered[0:kNN]:
                pred_score[0,i] = pred_score[0,i] + train[user_id,neigh] * similarity[i,neigh]
        rec_candidate = np.argsort(-pred_score)
        rec_candidate_X = rec_candidate[0]
        rec_list = rec_candidate_X[0:top_N]
        return rec_list

    def Evaluate(self,kNN,top_N):
        test = self.testdata
        num_users = self.num_users

        precision = 0
        recall = 0
        user_count = 0

        for i in np.arange(0,num_users):
            r_i = test[i]
            test_items = np.nonzero(r_i)
            test_items_idx = test_items[0]
            if len(test_items_idx) == 0:    # if this user does not possess rating in the test set, skip the evaluate procedure
                continue
            else:
                rec_of_i = self.Recommendation(i,kNN,top_N)
                hit_set = np.intersect1d(rec_of_i,test_items_idx)
                precision = precision + len(hit_set) / (top_N * 1.0)
                recall = recall + len(hit_set) / (len(test_items_idx) * 1.0)
                user_count = user_count + 1
        precision = precision / (user_count * 1.0)
        recall = recall / (user_count * 1.0)
        return precision,recall

def test():
    kNN = [80]
    top_N = [20]
    train = load_matrix('ua.base',943,1682)
    test = load_matrix('ua.test',943,1682)
    kNNItemCF = ItemCF(train,test)
    kNNItemCF.ItemSimilarity()
    print "%10s %10s %20s%20s" % ('kNN','top_N',"precision",'recall')
    for k in kNN:
        for N in top_N:
            precision,recall = kNNItemCF.Evaluate(k,N)
            print "%5d%5d%19.3f%%%19.3f%%" % (k,N,precision*100,recall*100)

if __name__=='__main__':
    test()

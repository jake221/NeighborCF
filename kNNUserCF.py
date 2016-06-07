# coding : UTF-8
# Please feel free to contact with me if you have any question with the code.
__author__ = 'wangjinkun@mail.hfut.edu.cn'

import numpy as np
import time

def load_matrix(filename,num_users,num_items):
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
    return matrix

class UserCF:

    def __init__(self,traindata,testdata):
        self.traindata = traindata
        self.testdata = testdata
        self.num_users = traindata.shape[0]
        self.num_items = traindata.shape[1]

    def UserSimilarity(self):
        t0 = time.time()
        train = self.traindata
        num_users = self.num_users
        self.user_similarity = np.zeros((num_users,num_users))

        for i in np.arange(0,num_users):
            r_i = train[i]
            self.user_similarity[i,i] = 0
            for j in np.arange(i+1,num_users):
                r_j = train[j]
                num = np.dot(r_i,r_j.T)
                denom = np.linalg.norm(r_i) * np.linalg.norm(r_j)
                if denom == 0:
                    cos = 0
                else:
                    cos = num / denom
                self.user_similarity[i,j] = cos
                self.user_similarity[j,i] = cos
        self.user_neighbor = np.argsort(-self.user_similarity)
        t1 = time.time()
        print 'Finished calculating similarity matrix in %f seconds' % (t1-t0)

    def Recommendation(self,user_id,kNN,top_N):
        # recommend a top_N recommendation list for user_id
        # r_ui = \sum_{v \in N^k(u)}r_vi \times w_uv
        train = self.traindata
        similarity = self.user_similarity

        # find the user's rating history
        r_u = train[user_id]
        rated_items = np.nonzero(r_u)
        rated_items_idx = rated_items[0]    # items rated by user_id in train set
        predict_items_idx = np.setdiff1d(np.arange(0,self.num_items),rated_items_idx)   # item index that has to be predicted
        pred_score = np.zeros((1,self.num_items))

        neighbor_ordered = self.user_neighbor[user_id]
        for i in predict_items_idx:
            item_idx = i
            for neigh in neighbor_ordered[0:kNN]:
                pred_score[0,i] = pred_score[0,i] + train[neigh,i] * similarity[user_id,neigh]
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
    top_N = [5,10,15,20]
    train_data = load_matrix('ua.base',943,1682)
    test_data = load_matrix('ua.test',943,1682)
    kNNUserCF = UserCF(train_data,test_data)
    kNNUserCF.UserSimilarity()
    print "%5s%5s%20s%20s" % ('kNN','top_N',"precision",'recall')
    for k in kNN:
        for N in top_N:
            precision,recall = kNNUserCF.Evaluate(k,N)
            print "%5d%5d%19.3f%%%19.3f%%" % (k,N,precision*100,recall*100)

if __name__=='__main__':
    test()
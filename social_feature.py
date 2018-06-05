import os
import numpy as np
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor

def load_social_features(video_id, video_user, user_details):
    vid = [] #video id list
    for line in open(video_id, encoding="utf-8"):
        vid.append(line.strip())
   
    vid_uid_dict = {} #vid-uid mapping
    for line in open(video_user, encoding="utf-8"):
        data = line.strip().split('::::')
        vid_uid_dict[data[0]] = data[1]
    
    social_features = {} #uid-social_feature mapping
    for line in open(user_details, encoding="utf-8"):
        data = line.strip().split("::::")
        # You should modify here to add more user social information
        #here we only use two user social infomation: loops and followers. You should consider more user social information. For more details about other social information, pls refer to ./data/README.txt -> 4.user_details.txt 
        social_features[data[0]] = [float(i) for i in data[1:6]]
        social_features.setdefault(data[0],[]).append(float(data[2])/float(data[3]))
        social_features.setdefault(data[0],[]).append(float(data[1])/float(data[2]))

    res = [] #social_feature vector for each video
    for v in vid:
        try:
            res.append(social_features[vid_uid_dict[v]])
        except:
            #note: there are some users don't have social features, just assgin zero-vector to them
            res.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) 

    return np.array(res, dtype=np.float32) 


def main():
    data_dir = './data/' 
    
    # load data
    print("Loading social feautres...")
    social_feature = load_social_features(data_dir + 'video_id.txt', data_dir + 'video_user.txt', data_dir + 'user_details.txt')
 
    concat_feature = social_feature
    print("The input data dimension is: (%d, %d)" %(concat_feature.shape))
    
    # load ground-truth
    ground_truth = []
    loops = []
    for line in open(os.path.join(data_dir, 'ground_truth.txt')):
        #you can use more than one popularity index as ground-truth and average the results; for each video we have four indexes: number of loops(view), likes, reposts, and comments; the first one(loops) is compulsory.
        truthlist = line.strip().split('::::')
        loops.append(float(truthlist[0]))
        ground_truth.append((float(truthlist[0]) + float(truthlist[1]) + float(truthlist[2]) + float(truthlist[3]))/4) 
    # ground_truth = np.array(ground_truth, dtype=np.float32)
    ground_truth = np.array(loops, dtype=np.float32)
    
    
    print("Start training and predict...")
    pred = np.empty(shape=[0,1])
    kf = KFold(n_splits=10)
    nMSEs = []
    for train, test in kf.split(concat_feature):
        # model initialize: you can tune the parameters within SVR(http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html); Or you can select other regression models
        # train
        # model = SVR()
        model = SVR(kernel='rbf', C=75000, gamma = 0.0001, epsilon = 0.01)
        # model = Ridge(alpha = 1.0)
        # model = linear_model.SGDRegressor()
        # model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.01, random_state=1)

        model.fit(concat_feature[train], ground_truth[train])
        # predict
        predicts = model.predict(concat_feature[test])
        pred = np.concatenate((pred, [[predict] for predict in predicts]))
        # nMSE(normalized Mean Squared Error) metric calculation
        nMSE = mean_squared_error(ground_truth[test], predicts) / np.mean(np.square(ground_truth[test]))
        nMSEs.append(nMSE)
        

        print("This round of nMSE is: %f" %(nMSE))
    
    print('Average nMSE is %f.' %(np.mean(nMSEs)))
    return pred


if __name__ == "__main__":
    main()

import os
import numpy as np
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
def main():
    data_dir = './data/' 
    
    # load data
    print("Loading image features...")
    hist_feature = np.load(data_dir + 'histogram_feature.npz')['arr_0']
    imgNet_feature = np.load(data_dir + 'imageNet_feature.npz')['arr_0']
    vSenti_feature = np.load(data_dir + 'visual_senti_feature.npz')['arr_0']
    sen2vec_feature = np.load(data_dir + 'text_sentence2vec_feature.npz')['arr_0']

    # feature dimension reduction: it's up to you to decide the size of reduced dimensions; the main purpose is to reduce the computation complexity
    pca = PCA(n_components=20)
    imgNet_feature = pca.fit_transform(imgNet_feature)
    pca = PCA(n_components=20)
    vSenti_feature = pca.fit_transform(vSenti_feature)
    pca = PCA(n_components=10)
    sen2vec_feature = pca.fit_transform(sen2vec_feature)
    pca = PCA(n_components=30)
    hist_feature = pca.fit_transform(hist_feature)
    # contatenate all the features(after dimension reduction)
    concat_feature = np.concatenate([hist_feature, imgNet_feature, vSenti_feature, sen2vec_feature], axis=1) 
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
        # model = SVR()
        model = SVR(kernel='rbf', C=75000, gamma = 0.0001, epsilon = 0.01)
        # model = linear_model.Ridge(alpha = 1.0)
        # model = GradientBoostingRegressor(max_depth =20, n_estimators=200, learning_rate=0.01, random_state=1)
        # model = RandomForestRegressor(max_depth=5, max_features='auto', min_samples_split=20, n_estimators=200, random_state=1)
        # train
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

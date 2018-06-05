import os
import numpy as np
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.model_selection import KFold, GridSearchCV

# grid search
# def svr_search(X, y, nfolds):
#     Cs = list(range(40000,120000, 10000))
#     gammas = [0.001, 0.005]
#     epsilon = [0.1, 0.01]
#     param_grid = {'C': Cs, 'gamma' : gammas, "epsilon":epsilon}
#     grid_search = GridSearchCV(SVR(kernel='rbf'), param_grid, cv=nfolds)
#     grid_search.fit(X, y)
#     grid_search.best_params_
#     return grid_search.best_params_


def main():
    data_dir = './data/' 
    
    # load data
    print("Loading textual feautres...")
    
    with open(os.path.join(data_dir, 'video_polarity.txt'), encoding='utf-8') as f:
        textual_feature = np.array([ [float(line.strip())] for line in f.readlines()])
    # feature dimension reduction: it's up to you to decide the size of reduced dimensions; the main purpose is to reduce the computation complexity
    concat_feature = textual_feature

    print("The input data dimension is: (%d, %d)" %(textual_feature.shape))
    
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
        model = SVR(kernel='rbf', C=75000, gamma = 0.0001, epsilon = 0.01)
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

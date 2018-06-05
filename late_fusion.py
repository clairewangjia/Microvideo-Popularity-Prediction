import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
import image_feature
import social_feature
import textual_feature

def main():
	data_dir = './data/'
	print("Loading data for late fusion model...")
	image_pred = np.array(image_feature.main())
	textual_pred = np.array(textual_feature.main())
	social_pred = np.array(social_feature.main())
	concat_predicts = np.concatenate([image_pred, textual_pred, social_pred], axis = 1)

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

	kf = KFold(n_splits=10)
	nMSEs = []

	for train, test in kf.split(concat_predicts):
		# model initialize: you can tune the parameters within SVR(http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html); Or you can select other regression models
		model = LinearRegression()
        # model = SVR()
		# model = SVR(kernel='rbf', C=80000, gamma = 0.0001, epsilon = 0.01)
        # model = linear_model.Ridge(alpha = 1.0)
        # model = GradientBoostingRegressor(max_depth =20, n_estimators=200, learning_rate=0.01, random_state=1)
        # model = RandomForestRegressor(max_depth=5, max_features='auto', min_samples_split=20, n_estimators=200, random_state=1)
		# train
		model.fit(concat_predicts[train], ground_truth[train])
		# predict
		predicts = model.predict(concat_predicts[test])
		# nMSE(normalized Mean Squared Error) metric calculation
		nMSE = mean_squared_error(ground_truth[test], predicts) / np.mean(np.square(ground_truth[test]))
		nMSEs.append(nMSE)

		print("This round of nMSE is: %f" %(nMSE))
    
	print('Average nMSE of late fusion model is %f.' %(np.mean(nMSEs)))

if __name__ == "__main__":
    main()
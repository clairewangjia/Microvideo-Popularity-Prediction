Lab 3: Viral Prediction
Deadline: 26 Mar 2018(Mon, 1800 Hrs)
Student: Wang Jia / A0176605B / E0232209

### Environment Setting
1. Windows 10
2. Python 3


### Installation 
0. emoji
1. nltk 
2. simplejson
3. pickle
4. numpy
5. scipy
6. scikit-learn
Note: please install emoji package for emoji description conversion via pip. (e.g., pip install emoji)  


### Usage 
1. Run 'early_fusion.py' to see early fusion results. (e.g. python early_fusion.py)
2. Run 'text_preprocess.py' to do preprocessing for description of micro-videos.
3. Run 'late_fusion.py' to see late fusion results.
Notes:
A. In the 'late_fusion.py' file, late fused SVR model will take several hours to get the result. Here I demonstrate the linear model results for faster viewing. SVR model can be run by commenting the line "model = SVR(...)".
B. 'image_feature.py', 'textual_feature.py' and 'social_feature.py' as 3 individual regressors will be called automatically when running 'late_fusion.py'. You are supposed to see performances (nMSE of each 10 fold and the average nMSE) printing on the screen for 3 individual models and the fused models:
	- image regressor
	- textual sentiment regressor
	- social classifier
	- late fused model
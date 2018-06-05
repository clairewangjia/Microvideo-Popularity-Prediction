### Introduction to Dataset
This dataset is a subset(10,000 videos) of the dataset(approximate 300,000 videos)[1] built with micro videos and attached information crawled from Vine(https://vine.co/). For each sample, it contains following information:
1. Image features extracted from one key frame image of original video, including color histogram feature, object features, visual sentiment features.
2. Text feature, inluding raw text attached with each video, a vector representation for each sentence.
3. Social feature of the videos's creator, including total loop count of the user, follower count, following count, like count, post count, etc.
4. Ground-truth, we use four indexes to represent the popularity of one video: number of loops, number of likes, number of reposts, and number of comments.


### Format of each file
1. video_id.txt: <videoid>
2. video_user.txt: <userid of the uploader of the video>
3. video_text.txt: <description of the video added by the uploader of the video> (some are empty)
4. user_details.txt:<userid>::::<total loop count of the user>::::<follower count of the user>::::<following count of the user>::::<like count of the user>::::<post count of the user>::::<twitter verified flag>::::<description of the user>::::<location of the user>::::<username of the user>
5. ground_truth.txt:<videoid>::::<number of loops>::::<number of likes>::::<number of reposts>::::<number of comments>::::<created time>
6. histogram_feature.npz: compressed numpy file, use numpy.load to load, is a 2-D array: (n, m), n is the number of samples; m is the dimension of histogram feature vector; here n=10000, m=50
7. imageNet_feature.npz: compressed numpy file, use numpy.load to load, is a 2-D array: (n, m), n is the n umber of samples; m is the dimension of imageNet feature vector; here n=10000, m=1000
8. text_sentence2vec_feature.npz: compressed numpy file, use numpy.load to load, is a 2-D array: (n, m), n is the number of samples; m is the dimension of sentence2vec feature vector; here n=10000, m=100
9. visual_senti_feature.npz: compressed numpy file, use numpy.load to load, is a 2-D array: (n, m), n is the number of samples; m is the dimension of visual sentiment feature vector; here n=10000, m=2059


### Feature extraction
Basically, all the extraction methods are refered to [2]. Feature extraction is not a part of this lab and you don't need to implement. Just provide for your information.
1. Image color histogram: using [opencv](https://opencv.org/)
2. Image object feature: using [resnet-50](https://github.com/KaimingHe/deep-residual-networks), model trained on ImageNet, on [Caffe](http://caffe.berkeleyvision.org/)
3. Visual sentiment feature: using [Visual Sentiment Ontology](http://visual-sentiment-ontology.appspot.com/)
4. Sentence2vec: using [sentence2vec](https://github.com/klb3713/sentence2vec)


### Please note that:
1.In the file of user_details.txt, '||||' means the corresponding attribute of the user does not exist.
2.Around 50 users' profile information can not be crawled since their accounts are protected.


### Citation
Please cite one or both of the following papers if you use the data in any way:
[1] J. Chen. Multi-modal learning: Study on A large-scale micro-video data collection.  In Proceedings of the 2016 ACM Conference on Multimedia Conference, MM 2016, Amsterdam, Netherlands, October 15-19, 2016, pages 1454–1458. ACM, 2016.
[2] J. Chen, X. Song, L. Nie, X. Wang, H. Zhang, and T. Chua. Micro tells macro: Predicting the popularity of micro-videos via a transductive model. In MM, pages 898–907. ACM, 2016.

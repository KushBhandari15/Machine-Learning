# Machine-Learning
ML libraries and working with Decision Trees and Random Forests

Download the data set: heart.csv; 
Watch Video on Random Forest Classification:
It is recommended that you follow this video tutorial on using sklearn. For the lab you should view the video between the 5:00 minute and 19:00 minute marks. The video will explain all the steps used in the starter code above, as well as showing you how to run an experiment with manual hyper-parameter tuning.

https://www.youtube.com/watch?v=BXkqEXjBf5s&ab_channel=MachineLearningLinks to an external site.

For the lab you will vary the number of default estimators for this classifier, as shown in the video. This tuning parameter is already named  n_estimators, and this parameter specifies the number of random trees used by the ensemble classifier. 

You will write code to search to find the number that gives the best accuracy on the test data, X_test. Report the scores for your results given by clf.score(X_test, Y_test) over a wide range of n_estimator values.

For the lab you should summarize your results in a concise data table. Finally, put this data table as a comment at the top of your code submission.

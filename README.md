# Automatic-Recognition-VAS-Index

This project contains the implementation of an automatic recognition system for pain perceived by a person. <br>
The goal of the system is therefore to analyze a sequence of frames in which the face of a subject is represented 
and then predict his perceived pain level based on the position of the subject's facial landmarks. 
The pain level is represented with an integer index between 0 and 10 called VAS (Visual Analog Scale).

<h2>Implementation</h2>
The project uses a dataset containing 200 sequences each made up of an arbitrary number of frames. 
Each frame within the videos is characterized by the position of 66 facial landmarks of the subject shown 
in the sequence and by the vas index corresponding to the pain perceived by it. <br>

Starting from the position of the landmarks detected in the frames, each sequence is described using fisher vectors, 
whose implementation is contained within the <b>fisherVector.py</b> script.
To apply this characterization, the positions of a subset of the landmarks of the various frames are clustered by training a Gaussian Mixture (GMM)
with a number of kernels defined a priori. 
This GMM is then used to describe the frames of each sequence of the dataset with the relative fisher vector: 
a multidimensional vector of dimensions equal to 2 * n * d with n number of kernels and d number of landmarks considered.
Each sequence is then described by a histogram obtained by adding the fisher vectors of the frames that compose it.<br>

The goal of the training procedure is to generate a model that predicts the vas index of a sequence starting from the histogram that describes it.
To accomplish this, a preliminary clustering phase is first performed and implemented in the <b>PreliminaryClustering.py</b> script.
This phase aims to extract the GMM kernels indices associated with the configurations considered relevant for the classification of the VAS index. 
Specifically, in the script those configurations of the landmarks that occur with a frequency greater than a certain threshold in a sequence of the dataset 
with associated vas index 0 are excluded. This allows to exclude configurations associated with a neutral expression, which in any case may occur in sequences with associated high pain.<br>

The results produced by this phase are input to the <b>ModelSVR.py</b> script, which is responsible for generating the model. To do this, each sequence of the dataset is described with the histogram obtained from the sum of only the relevant components of the fisher vectors associated with each frame of the video in question.
These descriptors are used for training a Support Vector Regression (SVR) able to predict the vas index of a sequence given the histogram that represents it. Specifically, the fitting involves the use of an RBF kernel and the parameters gamma, epsilon and C (regularization factor) are chosen in order to minimize the mean absolute error 
on the validation set data (using GridSearchCV of the sklearn library). Furthermore, if specific in input to the module, it is possible to train the model by weighing the examples based on the vas index associated with them. In other words, each sequence is associated with a weight inversely proportional to the number of occurrences of 
the examples in the training set having the same vas.
As for the calculation of performance, within ModelSVR.py there is also a method that calculates the mean absolute error and the confusion matrix obtained by predicting the vas associated with the sequences that compose the test set. 
The absolute error of a single prediction is calculated as the absolute value of the difference between real vas and predicted vas with predicted vas rounded to the nearest integer number. <br>

Note how the two modules described require as input to also receive the indices of the elements of the dataset to be used for this train procedure. The remaining indices will be used as test sets.

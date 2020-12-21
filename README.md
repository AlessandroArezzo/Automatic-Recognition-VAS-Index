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
a multidimensional vector of dimensions equal to 2 * n * d with n is the number of kernels and d is the number of landmarks considered.
Each sequence is then described by a histogram obtained by adding the fisher vectors of the frames that compose it.<br>

The goal of the training procedure is to generate a model that predicts the vas index of a sequence starting from the histogram that describes it.
To accomplish this, a preliminary clustering phase is first performed and implemented in the <b>PreliminaryClustering.py</b> script.
This phase aims to extract the GMM kernels indices associated with the configurations considered relevant for the classification of the VAS index. 
Specifically, in the script those configurations of the landmarks that occur with a frequency greater than a certain threshold in a sequence of the dataset 
with associated vas index 0 are excluded. This allows to exclude configurations associated with a neutral expression, which in any case may occur in sequences with associated high pain.<br>

The results produced by this phase are input to the <b>ModelSVR.py</b> script, which is responsible for generating the model. To do this, each sequence of the dataset is described with the histogram obtained from the sum of only the relevant components of the fisher vectors associated with each frame of the video.
These descriptors are used for training a Support Vector Regression (SVR) able to predict the vas index of a sequence given the histogram that represents it. Specifically, the fitting involves the use of an RBF kernel and the parameters gamma, epsilon and C (regularization factor) are chosen in order to minimize the mean absolute error 
on the validation set data (using GridSearchCV of the sklearn library). Furthermore, if specific in input to the module, it is possible to train the model by weighing the examples based on the vas index associated with them. In other words, each sequence is associated with a weight inversely proportional to the number of occurrences of 
the examples in the training set having the same vas.
As for the calculation of performance, within ModelSVR.py there is also a method that calculates the mean absolute error and the confusion matrix obtained by predicting the vas associated with the sequences that compose the test set. 
The absolute error of a single prediction is calculated as the absolute value of the difference between real vas and predicted vas with predicted vas rounded to the nearest integer number. <br>

Note how the two modules described require as input to also receive the indices of the elements of the dataset to be used for this train procedure. The remaining indices will be used as test sets.

<h2>How to use the code</h2>
The code can be used by running the following two scripts:
<ul>
  <li><b>test.py:</b> it allows to test and compare the performances of models obtained with a number of kernels fixed for the GMM by varying the threshold of the neutral configurations used in the preliminary clustering (see Implementation for more details).<br>
First, within the script the dataset is divided into multiple rounds, each characterized by a set of indices of the dataset that make up the train sequences and one that forms the validation sequences for the round. The subdivision occurs inherently in the validation protocol defined in the configuration file.<br>
Subsequently, for each threshold the training procedure is repeated for each round and for each of them the performance of the model on the respective validation set is calculated as the mean absolute error of the elements that compose it.
When this has been done for each round, the threshold performance is calculated as the average of the mean absolute errors of all rounds. At the end of the procedure for all the thresholds, a graph is saved showing the mean absolute errors as the thresholds used vary. Below is an example (obtained with 15 kernels for the GMM):
  <div align="center">
    <img src="/data/test/15_kernels/exp_weight_samples/errors_graph.png" width="300px"</img> 
  </div>
Furthermore, for each threshold a normalized confusion matrix is saved which allows to understand how the examples belonging to the various VAS index classes are predicted. Below is the example of the confusion matrix obtained with 15 kernels and a threshold equal to 0.35.
  <div align="center">
    <img src="/data/test/15_kernels/exp_weight_samples/confusion_matrices/confusion_matrix_0.35.png" width="300px"</img> 
  </div>
</li>
<li><b>generate_model_predictor.py:</b> it allows to perform the training of the model with a number of kernels of the GMM and a threshold of the neutral configurations defined a priori in the configuration file. As for the test.py module, here too the dataset is divided into rounds and for each of them the model is trained using the round's training set and then evaluated on its validation set.
For each round, a relative confusion matrix and a csv file that show the predictions detected for each element of its test set are generated.
At the end of the complete procedure, an overall confusion matrix (similar to the one shown in the image for the test.py script) and a bar graph showing the mean absolute errors detected at each round are saved. The graph obtained by running the script with 15 kernels and threshold equal to 0.35 is shown below:
  <div align="center">
    <img src="/data/classifier/15_kernels/graphics_errors.png" width="300px"</img> 
  </div>
</li>
</ul>

<h2>How to set the parameters</h2>
The script parameters can be set in the <b>config.py</b> file inside the configuration directory of the project.
The parameters that can be specified are the following:
<ul>
  <li><b>n_kernels_GMM:</b> it defines the number of kernels to be used for the Gaussian Mixture in the preliminary clustering phase. </li>
  <li><b>selected_lndks_idx:</b> it specifies the indexes of the landmarks to be considered during the procedure.</li>
  <li><b>n_jobs:</b> number of threads to use to perform SVR training.</li>
  <li><b>cross_val_protocol:</b> type of protocol to be used to evaluate the performance of the models. The following three protocol values are permitted:
  'Leave-One-Subject-Out', '5-fold-cross-validation' and 'Leave-One-Sequence-Out'.</li>
  <li><b>weighted_samples:</b> it defines if the samples must be weighted for the SVR training (see Implementation for more details).</li>
  <li><b>threshold_neutral:</b> it defines the value to use as threshold for neutral configurations in the generate_model_predictor.py script.</li>
  <li><b>thresholds_neutral_to_test:</b> it defines the range of threshold values to be used in the test.py script.</li>
  <li><b>save_histo_figures:</b> it defines if the histograms of the dataset sequences must be saved in the generate_model_predictor.py.</li>
</ul>

<h2>Where are the results</h2>
The results generated by the test.py script are saved in the project in the data / test / n_kernels / directory (with n is the number of kernels of the GMM used for the experiments).<br>
The files saved in output by the generate_model_predictor.py script are saved in the data / classifier / n_kernels directory (with n indicating also in this case the number of kernels of the GMM).

<h2>Prerequisites</h2>
To use the code you need to have the following libraries installed:
<ul>
  <li>scikit-learn v.0.23.1</li>
  <li>pandas v.1.0.5</li>
  <li>matplotlib v.3.2.2</li>
  <li>numpy v.1.19.0</li>
  <li>seaborn v.0.11.0</li>
</ul>

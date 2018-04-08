# Semi_Supervised_Auto_Encoder

The Python script Semi_Supervised_Auto_Encoder implements the model described in [1]
The implementaiton exploits the KERAS library for Deep Learning (https://keras.io) and it also employs utilities from the Scikit-Learn library (http://scikit-learn.org/stable).
It takes as input 3 parameters:
1) An .npy file (data file) containing the data (examples and characteristics)
2) An .npy file (class file) containing as many row as the number of training examples (examples with class label) and two columns: the id of the training example (it indicates the position of the sample in the first data file) and the class value associated to that example.
3) The ensemble size in order to generate the new (semi-supervised) embedding representation. 

In this github page, an example of datasets (data.npy) and class file (0_10.npy) is supplied. The dataset is the .npy representation of the sonar data public available at the UCI Machine Learning Repository (http://tunedit.org/repo/UCI/sonar.arff).
As previously mentioned, the program needs both KERAS and SCIKIT-LEARN python library installed.

The command line to run the program is the following one:
  python semi_supervised_auto_encoder.py data.npy 0_10.npy 10

The result will be stored in the file "representation.npy" that will contain as many example as the number of example in the data.npy file.

[1] D. Ienco and R. G. Pensa: "Semi-Supervised Clustering with Multiresolution Autoencoders". International Joint Conference on Neural Network (IJCNN) 2018.

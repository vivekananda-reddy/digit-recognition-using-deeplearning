# Digit Recognition Using Deeplearning
This reposetory contains a deep learning neural network model for recognising hand written digits.
The core implementation is done in python using Tensor flow.

# Libraries/dependencies

1. numpy
2. pandas
3. tensorflow
4. math
5. pickle
6. PIL
7. tkinter

# Description of the code files
1. data_processing.py: In this file data is preprocessed. The data from train and test CSV files is loaded using padas and then converted into the numpy arrays.
2. ml_using_tensorflow.py: In this file the deep neural network is being implemented from the data preprossed in data_processing.py file
3. UI_for_digitpredictor.py: In this file a simple UI is build with in tkinter with 2 buttons -- train and predict. Train button is used to train the network and predict button will read the input handwrittem digit image and give the predicted value.
4. Source data set can be downloaded from https://www.kaggle.com/oddrationale/mnist-in-csv



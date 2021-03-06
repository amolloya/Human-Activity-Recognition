# Human-Activity-Recognition

In this section, we will use the machine learning techniques like PCA and CNN for a typical Human Activity Recognition problem and implement it in Python with keras library. </br>
* Principal Component Analysis (PCA) is a dimensionality reduction techniques which makes the high-dimensional input data into low-dimensional one while maintaining maximum variance. </br>
* Convolutional neural networks (CNN) are deep artificial neural networks that are used primarily to classify images, cluster them by similarity (image search on google), perform object recognition within scenes or task recognition.

The problem we try to solve here is of Human Activity Recognition (HAR) task for classifying the walking trajectories of individuals into different speeds (slow, comfortable or fast). We use PCA and CNN models for HAR classification task. </br> The input to our model is the walking trajectory (x,y,z co-ordinates) of individuals, which is a time-series of 66-marker positions with respect to time, and the output is the speed class (slow, comfartable or fast) that the individual is walking at.

The data for this problem is taken from a public dataset available at: https://figshare.com/articles/A_public_data_set_of_overground_and_treadmill_walking_kinematics_and_kinetics_of_healthy_individuals/5722711
</br> The data is loaded, preprocessed and filtered in the files: WBDSascii_data and WBDSascii_data3.

For the first model, we first import the data from the WBDSascii_data file and using keras we develop a 3-layered CNN model. This model gives us an accuracy of over 96%. The code for this model is in the file CNN.py.

For the other model, we first import the data from the WBDSascii_data3 file, which uses PCA and reduces the dimension of the data by a factor of 1/10th saving us a lot of computational power (which increases exponentially with the size of the data) and drastically reducting the time required to train the model. After that using keras we develop a 3-layered CNN model. This model gives us an accuracy of over 90%. Note that PCA is carried out in the file (WBDSascii_data3) and not in the main file (PCA_CNN.py). The code for this model is in the file PCA_CNN.py.

These two models gives the most general trade-off a machine learning analyst has to consider while solving a problem with machine learning. The trade-off being between accuracy and computational power. So depending upon what is important for your particular application, we choose the appropriate model. </br> Note that, we have kept the hyper-parameters (like number of epochs, batch size, number of convolutional layers, number of filters) same for both the models to compare the results obtained. 

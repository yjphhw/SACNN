# More will update soon
# SACNN (Spatial Adversarial Convolutional Neural Network for Surface Defect Detection)
Constructing surface defect detection systems are significant to quality control in industrial production, but it is costly and laborious to label sufficient detailed samples. This paper proposes a model called spatial adversarial convolutional neural network (SACNN) which only uses image level label for detect surface defect. It consists of two parts: feature extractor and feature competition. Firstly, a string of convolutional blocks is used as feature extractor. After feature extraction, the maximum greedy feature competition is taken among features in the feature layer. During training, the model can spontaneously focus to the actual defective position, and is robust to sample imbalance. The classification accuracy of the two datasets can reach more than 98%, and is comparable with to the method of labeling the samples in detail. Detection results show that the defect location is more compact and accurate than Grad-CAM method. Experiments show that our model has potential usage in defect detection under industrial environment.

![Image text](https://raw.githubusercontent.com/yjphhw/SACNN/master/nnstructure2.png "Structure of SACNN")



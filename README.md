# CV Remix
a collection of wonderful models and toy projects.

## RGB-D object recognition
Spent several hours to reproduce this paper: Multimodal Deep Learning for Robust RGB-D Object Recognition. The principle work is to train two AlexNet and concat the top FC layers. One sub-graph has to be freezed while training the another one.  
I cannot directly train the model since the tensorflow I compiled doesn't contain tf.contrib package. My friend says the performance on a modified RGB-d object dataset is good after fixing two bugs.

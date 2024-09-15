# Breast-cancer-detection-CNN
This repository includes a small python code where a CNN was trained on a small dataset containing images classified in 8 different categories; 4 of them representing Benign tumors and the other 4 malignant.



# Motivation
Breast cancer is one of the most common cancer types suffered by women nowadays. In 2022 there were more than 2.3 million women diagnosed with it; and around 650 thousand deaths globally. This implied that any kind of mechanism we can construct to identify it early enough to start treatment would be a great advance. The most used technique for this is the use of mammographies (X-Ray image of the breast); which give us an image of the interior of the breast to see if there exists any harmful tumor. Sometimes this presence detection is not so straight-forward; so using Neural Networks could be a good solution; at least as a first filter so that real doctors can confirm or analyze in depth the most important results.

# Breast density
The images used basically show breast density. If the breast contains high levels of fat; its density will be lower and any type of tumor (higher density) would be identified easier. However, is the breast has a noticeable density; the identification of a tumor may get harder. Moreover, women with denser breasts have a higher chance to get breast cancer than those with fatty breats; so it is definitely an important factor in breast cancer detection/diagnosys.

# The Dataset
The dataset used here was taken from another GitHub repository (Breast-Caner-Detection) and contains about 5000 mammographies to be used as training/validation data (labeled) and about 1800 test images (which are NOT labeled). All images are in a (224,224,3) format. The labels used run from Density1 to Density4; increasing the breasts density as the number increases. They are also classified as Benign or Malignant.

Here you have some examples of the 8 different categories:
![D1B](https://github.com/user-attachments/assets/5c41e9a7-4a68-46a5-94dd-e11e1e3c0954)
![D2B](https://github.com/user-attachments/assets/44ea3b70-ebd9-47ef-aa6a-6dc5ce2af917)
![D3B](https://github.com/user-attachments/assets/a30a2363-1414-4752-b44d-53b7b4a6a56e)
![D4B](https://github.com/user-attachments/assets/7c72efc1-5af8-4b08-84b1-2e7810f39d82)
![D1M](https://github.com/user-attachments/assets/9bd1d2a5-f5bd-4131-b53a-094b81a63d9a)
![D2M](https://github.com/user-attachments/assets/a96b9541-61ed-4d7b-842a-b535eceb9a69)
![D3M](https://github.com/user-attachments/assets/7d354770-c541-4dcc-a08c-ef9134f0dfde)
![D4M](https://github.com/user-attachments/assets/923cb718-fdee-499f-9803-655ecfd5c4d7)

# Image preprocessing
Before training the NN, we had a look at our images to see if they were detailed enough or if we could do some changes to them. We tried three different filters to convolve them with; and then chose the one that seemed to improve the image the most.
<br> 
We used a Gaussian Kernel:

$$ {\left\lbrack \matrix{1/16 & 1/8 & 1/16 \cr 1/8 & 1/4 & 1/8 \cr 1/16 & 1/8 & 1/16} \right\rbrack} $$

A Mean Kernel:

$$ {\left\lbrack \matrix{1/9 & 1/9 & 1/9 \cr 1/9 & 1/9 & 1/9 \cr 1/9 & 1/9 & 1/9} \right\rbrack} $$

And a Laplacian (not normalized) kernel:

$$ {\left\lbrack \matrix{-1 & -1 & -1 \cr -1 & 9 & -1 \cr -1 & -1 & -1} \right\rbrack} $$

The results after applying these three filters to the images were the following:

Original mammography:

![normal](https://github.com/user-attachments/assets/51e666b8-06ed-4f8b-bc72-ed57f255397e)


Gaussian Kernel:

![gaussian](https://github.com/user-attachments/assets/a0850870-b37d-4397-9959-cdf7dff6d985)


Mean Kernel:

![mean](https://github.com/user-attachments/assets/d80ca890-2461-4e81-9eee-4650c9b0bc5b)


Laplacian Kernel:

![laplacian](https://github.com/user-attachments/assets/5209b201-6034-4353-a15a-f44e9ba21c3f)

As it is easily seen, the last filter increases the contrast on the image and thus makes it better to use; so this will be the selected filter. A part from this no other preprocessing was conducted.

# Neural Network architecture

The Neural Network used is completely new (meaning it does not use transfer learning). It has three convolutional layers where the kernel size is 3x3 (based on the VGG network architecture), each of them followed by a Batch Normalization layer and pooling layer (MaxPooling of 2x2). After the three group of layers, we flatten the network and include three Dense Layers (with dropout layers as well) that end up in a 8-unit softmax layer; which will tell us the category that has been chosen to classify the specific image. Inside the .ipynb file there is an explanation deriving the number of parametrs in each layer. The model was fitted during 45 epochs (due to its relativily large size for a normal portable computer, we could not train for much more). 
\n
It is important to stand out that, although this specific project does not use transfer learning as a trial to see how weel could my own computer do classifying the images; Transfer Learning is a really powerful tool that can and should be used when dealing with real world applications (using for example the previously mentioned VGG network or others such as DenseNet201).

# Results





















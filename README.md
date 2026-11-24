# Fashion MNIST baseline
## What it is
- Classifies images into 10 categories by learning weights from their pixel values. 
- The images are 28 x 28 greyscale pictures of clothing from the Fashion MNIST dataset. 
- Reaches 86.14% test accuracy.

## Data
- The dataset has 60000 training images and 10000 test images. 
- 10 classes are T-shirt,trouser,pullover,dress,coat,sandal,shirt,sneaker,bag and ankle boot. 
- It was loaded in from torchvision with pixels scaled from 0-1.

## Model
- **Flatten**: Turns each 28x28 image into a row of 784 numbers
- **Linear(784,128)**: 128 weighted sums over the pixels where each one is a pattern that the model learns
- **ReLU**: Any negative numbers are set to 0 for the model to be able to learn curved patterns 
- **Linear(128,10)** : One score per class and the biggest score is set to be the prediction

## Training
- Cross entropy loss: converts the 10 scores into probabilities and measures how far the probability of the correct class is from 1
- Optimiser - Adam, learning rate of 0.001
- Batch size is 64 with 3 epochs and training data being shuffled 
- The training loss fell from about 0.55 to 0.36

## Results
- Test accuracy of 0.8614 on the 10000 images the model never trained on
- The accuracy is the share of images where the biggest score is the correct class 

## Limitations 
- No validation set so nothing was tuned and the test set was used once 
- The fully connected layers ignore the 2d structure of the image

## Next steps 
- add a val split, tune epochs, tune hidden size, try a CNN

## HOW TO RUN 
- `pip install torch torchvision`, then `python train.py`



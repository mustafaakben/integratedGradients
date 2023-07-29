# Integrated Gradients Method for Sentiment Analysis

This repository provides a simple and understandable implementation of the integrated gradients method for sentiment analysis on the IMDB movie reviews dataset.

The integrated gradients method is a technique for attributing the prediction of a deep learning model to its input features. It helps me understand the importance of each feature in making the prediction. 

In this case, I use it to understand which words in a movie review contribute most significantly towards its sentiment (positive or negative). This can be invaluable for understanding and interpreting my model's predictions.

## Methodology

1. I load the IMDB movie reviews dataset using Keras's built-in utilities. The dataset is split into training and testing sets.

2. Next, I preprocess the text data. This includes converting the reviews into a form that can be used by the model, i.e., one-hot encoding.

3. Then, I construct a custom embedding layer using Einstein Summation to allow for gradients to pass through, enabling me to obtain gradients with respect to the inputs.

4. A simple neural network model is built, trained on the training set, and evaluated on the test set.

5. After the model is trained, I use the integrated gradients method to compute attributions. The integrated gradients are calculated for the difference between the baseline (a tensor of zeros) and the input instance.

6. Finally, I retrieve the most important words contributing to the sentiment of a review, providing an interpretation of the model's prediction.

## Setup and Usage

To run this code, you need to have Python and the following packages installed: TensorFlow, NumPy, and Keras.

## Custom Embedding Layer

A key part of this implementation is the custom embedding layer. Many standard embedding layers, including those provided by TensorFlow, do not allow gradients to pass through them. This is because they use lookup or gather functions under the hood, which block the gradients. To get around this issue, I implemented a custom embedding layer using Einstein Summation, which allows for the gradients to pass through.

Please note, the custom embedding layer requires that the inputs are one-hot encoded, which is done during the preprocessing step.

Happy coding!

# A Simple Implementation of Integrated Gradient Method

This is is a simple integrated gradient implementation. In this example, I trained a simple deep learning model with IMDB movie review dataset. One of the major bottleneck to get the gradient w.r.t. inputs was that embedding layers do not let the input gradients pass as they use lookup or gather functions. 

To overcome this issue, I wrote a custom embedding layer with Einstein Summation; for this custom layer to work, inputs should be one-hot encoded text.
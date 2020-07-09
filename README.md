
# ParameterEffectsonNueralNetworks

In this Project, We will analyze MLPClassifier in Scikit-learn Library and investigate
the effect of different choice of activation functions, values of regularization parameters,
numbers of hidden units (single hidden layer), numbers of hidden layers separately.

## Result Analysing
According to these results, the model with 100 hidden units in 1 layer, using 5.0
alpha degree and ReLu activation function gets the best score.
### Why ReLu function is the best parameter?
-The main advantage of using the ReLU function over other activation functions is
that it does not activate all the neurons at the same time. This means that the neurons will
only be deactivated if the output of the linear transformation is less than 0.[2] In this dataset
we have two dimensional matrices that include 0 and 1. Other functions could get active the
model with negative numbers (Noisy data) and it can be harmful for our score. For this
reason, I think ReLu activation function might be more successful than other functions.

### Why 1 layer and 100 units got best score?
One of ReLU's benefit is sparsity. Sparsity arises when 𝑎≤0. The more such units that exist
in a layer the more sparse the resulting representation. Sigmoids on the other hand are
always likely to generate some non-zero value resulting in dense representations. Sparse
representations seem to be more beneficial than dense representations.[3] In this way, we
can say that more units will add less sparity to our model.

## References
https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html [2]
https://www.analyticsvidhya.com/blog/2020/01/fundamentals-deep-learning-activation-functions-when-to-use-them/ [3]
https://stats.stackexchange.com/questions/126238/what-are-the-advantages-of-relu-over-sigmoid-function-in-deep-neural-networks

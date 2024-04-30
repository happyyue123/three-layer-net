"""
Implements a two-layer Neural Network classifier in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""
import torch
import random
import statistics
import numpy as np
from typing import Dict, List, Callable, Optional


def hello_three_layer_net():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print("Hello from two_layer_net.py!")


# Template class modules that we will use later: Do not edit/modify this class
class ThreeLayerNet(object):
    def __init__(
        self,
        input_size: int,
        hidden1_size: int,
        hidden2_size: int,
        output_size: int,
        dtype: np.float32 = np.float32,
        std: float = 1e-4,
    ):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)
        W2: Third layer weights; has shape (C, S)
        b2: Third layer biases; has shape (S,)

        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes S.
        - dtype: Optional, data type of each initial weight params
        - device: Optional, whether the weight params is on GPU or CPU
        - std: Optional, initial weight scaler.
        """
        # reset seed before start
        np.random.seed(0)

        self.params = {
            'W1': np.random.randn(input_size, hidden1_size) * 0.01,
            'b1': np.zeros(hidden1_size),
            'W2': np.random.randn(hidden1_size, hidden2_size) * 0.01,
            'b2': np.zeros(hidden2_size),
            'W3': np.random.randn(hidden2_size, output_size) * 0.01,
            'b3': np.zeros(output_size)}
    def loss(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        reg: float = 0.0,
    ):
        return nn_forward_backward(self.params, X, y, reg)

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        learning_rate: float = 1e-3,
        learning_rate_decay: float = 0.95,
        reg: float = 5e-6,
        num_iters: int = 100,
        batch_size: int = 200,
        verbose: bool = False,
    ):
        # fmt: off
        return nn_train(
            self.params, nn_forward_backward, nn_predict, X, y,
            learning_rate, learning_rate_decay,
            reg, num_iters, batch_size, verbose,
        )
        # fmt: on

    def predict(self, X: np.ndarray):
        return nn_predict(self.params, nn_forward_backward, X)

    def save(self, path):
        # 使用 numpy.savez 保存参数字典中的所有数组
        np.savez(path, **self.params)
        print("Saved in {}".format(path))

    def load(self, path):
        # 使用 numpy.load 加载所有数组
        data = np.load(path)
        # 确保加载了正确的参数数量和具体的参数名
        if len(data.keys()) != 4:
            raise Exception("Failed to load your checkpoint")
        for param in ["W1", "b1", "W2", "b2", "W3", "b3"]:
            if param not in data:
                raise Exception("Failed to load your checkpoint")
        # 赋值回 self.params
        self.params = {key: data[key] for key in data.files}
        print("Loaded checkpoint from: {}".format(path))


def nn_forward_pass(params: Dict[str, torch.Tensor], X: torch.Tensor):
    """
    The first stage of our neural network implementation: Run the forward pass
    of the network to compute the hidden layer features and classification
    scores. The network architecture should be:

    FC layer -> ReLU (hidden1) -> FC layer -> ReLU(hidden2) -> FC layer (scores)

    As a practice, we will NOT allow to use torch.relu and torch.nn ops
    just for this time (you can use it from A3).

    Inputs:
    - params: a dictionary of PyTorch Tensor that store the weights of a model.
      It should have following keys with shape
          W1: First layer weights; has shape (D, H)
          b1: First layer biases; has shape (H,)
          W2: Second layer weights; has shape (H, C)
          b2: Second layer biases; has shape (C,)
          W3: Third layer weights; has shape(C, S)
          b3: Third layer biases; has shape(S,)
    - X: Input data of shape (N, D). Each X[i] is a training sample.

    Returns a tuple of:
    - scores: Tensor of shape (N, S) giving the classification scores for X
    - hidden1: Tensor of shape (N, H) giving the hidden layer representation
    - hidden2: Tensor of shape (N, C) giving the hidden layer representation
      for each input value (after the ReLU).
    """
    # Unpack variables from the params dictionary
    W1, b1 = params["W1"], params["b1"]
    W2, b2 = params["W2"], params["b2"]
    W3, b3 = params["W3"], params["b3"]
    N, D = X.shape

    # Compute the forward pass
    hidden1 = None
    hidden2 = None
    scores = None
    ############################################################################
    # TODO: Perform the forward pass, computing the class scores for the input.#
    # Store the result in the scores variable, which should be an tensor of    #
    # shape (N, C).                                                            #
    ############################################################################
    # Replace "pass" statement with your code
    s1 = np.dot(X, W1) + b1
    # 应用ReLU激活函数
    hidden1 = np.maximum(s1, 0)  # np.maximum用于计算逐元素的最大值，0是ReLU的阈值

    # 第二层的线性变换
    s2 = np.dot(hidden1, W2) + b2
    # 应用ReLU激活函数
    hidden2 = np.maximum(s2, 0)

    # 输出层的线性变换
    scores = np.dot(hidden2, W3) + b3
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return scores, hidden1, hidden2


def nn_forward_backward(
    params: Dict[str, torch.Tensor],
    X: torch.Tensor,
    y: Optional[torch.Tensor] = None,
    reg: float = 0.0
):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network. When you implement loss and gradient, please don't forget to
    scale the losses/gradients by the batch size.

    Inputs: First two parameters (params, X) are same as nn_forward_pass
    - params: a dictionary of PyTorch Tensor that store the weights of a model.
      It should have following keys with shape
          W1: First layer weights; has shape (D, H)
          b1: First layer biases; has shape (H,)
          W2: Second layer weights; has shape (H, C)
          b2: Second layer biases; has shape (C,)
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a tensor scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = params["W1"], params["b1"]
    W2, b2 = params["W2"], params["b2"]
    W3, b3 = params["W3"], params["b3"]
    N, D = X.shape

    scores, h1, h2 = nn_forward_pass(params, X)
    # If the targets are not given then jump out, we're done
    if y is None:
        return scores

    # Compute the loss
    loss = None
    ############################################################################
    # TODO: Compute the loss, based on the results from nn_forward_pass.       #
    # This should include both the data loss and L2 regularization for W1 and  #
    # W2. Store the result in the variable loss, which should be a scalar. Use #
    # the Softmax classifier loss. When you implment the regularization over W,#
    # please DO NOT multiply the regularization term by 1/2 (no coefficient).  #
    # If you are not careful here, it is easy to run into numeric instability  #
    # (Check Numeric Stability in http://cs231n.github.io/linear-classify/).   #
    ############################################################################
    # Replace "pass" statement with your code
    # 计算loss，每个类别的概率，Li = -log（p正确类别) L = sum(Li) + lambda(W1^2 + W2^2)
    # 数值稳定性：从得分中减去每行的最大值
    # 计算 softmax 概率
    shifted_scores = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(shifted_scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    epsilon = 1e-10  # 小常数以防止对零取对数
    correct_logprobs = -np.log(probs[np.arange(N), y] + epsilon)
    data_loss = np.mean(correct_logprobs)
    reg_loss = reg * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2))
    loss = data_loss + reg_loss

    # Backward pass
    dscores = probs
    dscores[np.arange(N), y] -= 1
    dscores /= N

    # W3 and b3 gradients
    dW3 = h2.T.dot(dscores) + 2 * reg * W3
    db3 = np.sum(dscores, axis=0)

    # Backprop into second layer
    dh2 = dscores.dot(W3.T)
    dh2[h2 <= 0] = 0

    # W2 and b2 gradients
    dW2 = h1.T.dot(dh2) + 2 * reg * W2
    db2 = np.sum(dh2, axis=0)

    # Backprop into first layer
    dh1 = dh2.dot(W2.T)
    dh1[h1 <= 0] = 0

    # W1 and b1 gradients
    dW1 = X.T.dot(dh1) + 2 * reg * W1
    db1 = np.sum(dh1, axis=0)

    # Store gradients
    grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2, 'W3': dW3, 'b3': db3}

    return loss, grads

def sample_batch(
  X:np.ndarray, y: np.ndarray, num_train: int, batch_size:int
):
  """
  Sample batch_size elements from the training data and 
  their corresponding labels to use in this round of gradinet decent
  """
  indice = np.random.randint(0, num_train, size=batch_size)

  X_batch = X[indice]
  y_batch = y[indice]

  return X_batch, y_batch
def nn_train(
    params: Dict[str, np.ndarray],
    loss_func: Callable,
    pred_func: Callable,
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 1e-3,
    learning_rate_decay: float = 0.95,
    reg: float = 5e-6,
    num_iters: int = 100,
    batch_size: int = 200,
    verbose: bool = False,
):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - params: a dictionary of PyTorch Tensor that store the weights of a model.
      It should have following keys with shape
          W1: First layer weights; has shape (D, H)
          b1: First layer biases; has shape (H,)
          W2: Second layer weights; has shape (H, C)
          b2: Second layer biases; has shape (C,)
    - loss_func: a loss function that computes the loss and the gradients.
      It takes as input:
      - params: Same as input to nn_train
      - X_batch: A minibatch of inputs of shape (B, D)
      - y_batch: Ground-truth labels for X_batch
      - reg: Same as input to nn_train
      And it returns a tuple of:
        - loss: Scalar giving the loss on the minibatch
        - grads: Dictionary mapping parameter names to gradients of the loss with
          respect to the corresponding parameter.
    - pred_func: prediction function that im
    - X: A PyTorch tensor of shape (N, D) giving training data.
    - y: A PyTorch tensor of shape (N,) giving training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - X_val: A PyTorch tensor of shape (N_val, D) giving validation data.
    - y_val: A PyTorch tensor of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.

    Returns: A dictionary giving statistics about the training process
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train // batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []

    for it in range(num_iters):
        
        X_batch, y_batch = sample_batch(X, y, num_train, batch_size)
        loss, grads = loss_func(params, X_batch, y_batch, reg)
        loss_history.append(loss)

        #########################################################################
        # TODO: Use the gradients in the grads dictionary to update the         #
        # parameters of the network (stored in the dictionary self.params)      #
        # using stochastic gradient descent. You'll need to use the gradients   #
        # stored in the grads dictionary defined above.                         #
        #########################################################################
        # Replace "pass" statement with your code
        for param in params:
          params[param] = params[param] - grads[param] * learning_rate


        
        #########################################################################
        #                             END OF YOUR CODE                          #
        #########################################################################

        if verbose and it % 100 == 0:
            print(f"iteration {it} / {num_iters}: loss {loss:.3f}")

        # Every epoch, check train and val accuracy and decay learning rate.
        if it % iterations_per_epoch == 0:
            # Check accuracy
            y_train_pred = pred_func(params, loss_func, X_batch)
            train_acc = np.mean(y_train_pred == y_batch)
            
            train_acc_history.append(train_acc)
            
            # Decay learning ratey'r
            learning_rate *= learning_rate_decay

    return {
        "loss_history": loss_history,
        "train_acc_history": train_acc_history,
        
    }


def nn_predict(
    params: Dict[str, torch.Tensor], loss_func: Callable, X: torch.Tensor
):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - params: a dictionary of PyTorch Tensor that store the weights of a model.
      It should have following keys with shape
          W1: First layer weights; has shape (D, H)
          b1: First layer biases; has shape (H,)
          W2: Second layer weights; has shape (H, C)
          b2: Second layer biases; has shape (C,)
    - loss_func: a loss function that computes the loss and the gradients
    - X: A PyTorch tensor of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A PyTorch tensor of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    y_pred = None

    ###########################################################################
    # TODO: Implement this function; it should be VERY simple!                #
    ###########################################################################
    # Replace "pass" statement with your code
    scores, h1, h2 = nn_forward_pass(params, X)
    y_pred = np.argmax(scores, axis=1)

    return y_pred


def nn_get_search_params():
    """
    Return candidate hyperparameters for a TwoLayerNet model.
    You should provide at least two param for each, and total grid search
    combinations should be less than 256. If not, it will take
    too much time to train on such hyperparameter combinations.

    Returns:
    - learning_rates: learning rate candidates, e.g. [1e-3, 1e-2, ...]
    - hidden_sizes: hidden value sizes, e.g. [8, 16, ...]
    - regularization_strengths: regularization strengths candidates
                                e.g. [1e0, 1e1, ...]
    - learning_rate_decays: learning rate decay candidates
                                e.g. [1.0, 0.95, ...]
    """
    learning_rates = []
    hidden_sizes = []
    regularization_strengths = []
    learning_rate_decays = []
    ###########################################################################
    # TODO: Add your own hyper parameter lists. This should be similar to the #
    # hyperparameters that you used for the SVM, but you may need to select   #
    # different hyperparameters to achieve good performance with the softmax  #
    # classifier.                                                             #
    ###########################################################################
    # Replace "pass" statement with your code
    pass
    ###########################################################################
    #                           END OF YOUR CODE                              #
    ###########################################################################

    return (
        learning_rates,
        hidden_sizes,
        regularization_strengths,
        learning_rate_decays,
    )


def find_best_net(
    data_dict: Dict[str, torch.Tensor], get_param_set_fn: Callable
):
    """
    Tune hyperparameters using the validation set.
    Store your best trained TwoLayerNet model in best_net, with the return value
    of ".train()" operation in best_stat and the validation accuracy of the
    trained best model in best_val_acc. Your hyperparameters should be received
    from in nn_get_search_params

    Inputs:
    - data_dict (dict): a dictionary that includes
                        ['X_train', 'y_train', 'X_val', 'y_val']
                        as the keys for training a classifier
    - get_param_set_fn (function): A function that provides the hyperparameters
                                   (e.g., nn_get_search_params)
                                   that gives (learning_rates, hidden_sizes,
                                   regularization_strengths, learning_rate_decays)
                                   You should get hyperparameters from
                                   get_param_set_fn.

    Returns:
    - best_net (instance): a trained TwoLayerNet instances with
                           (['X_train', 'y_train'], batch_size, learning_rate,
                           learning_rate_decay, reg)
                           for num_iter times.
    - best_stat (dict): return value of "best_net.train()" operation
    - best_val_acc (float): validation accuracy of the best_net
    """

    best_net = None
    best_stat = None
    best_val_acc = 0.0

    #############################################################################
    # TODO: Tune hyperparameters using the validation set. Store your best      #
    # trained model in best_net.                                                #
    #                                                                           #
    # To help debug your network, it may help to use visualizations similar to  #
    # the ones we used above; these visualizations will have significant        #
    # qualitative differences from the ones we saw above for the poorly tuned   #
    # network.                                                                  #
    #                                                                           #
    # Tweaking hyperparameters by hand can be fun, but you might find it useful #
    # to write code to sweep through possible combinations of hyperparameters   #
    # automatically like we did on the previous exercises.                      #
    #############################################################################
    # Replace "pass" statement with your code
    pass
    #############################################################################
    #                               END OF YOUR CODE                            #
    #############################################################################

    return best_net, best_stat, best_val_acc

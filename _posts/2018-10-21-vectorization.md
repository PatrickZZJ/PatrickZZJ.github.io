---
layout: post
title: '向量化编程总结（Vectorization）'
date: 2018-10-21
author: Patrick Zhang
color: rgb(16, 46, 94)
cover: '/assets/vectorization.jpeg'
tags: coding python
---

> 向量化编程能极大地加快程序运行速度，是机器学习、数据处理等领域的重要技术，在此文中通过例子总结一下向量化编程的思想和方法

### 一. 为什么向量化编程

 	向量化编程可理解为将基于循环、标量的代码修改为使用矩阵和向量运算的过程。使用向量化编程可以更好地利用硬件提供的并行性。具体的来说**向量化（vectorization）**是为了开发**数据级并行（data-level parallelism）**，这通过处理器支持**SIMD（single instruction, multiple data）**指令来完成。SIMD的优点是所有并行执行的单元都是同步的，他们都对源于同一程序计数器（PC）的同一指令作出相应，这样可以在执行单元之间均摊控制成本提供并行，还可以降低指令的宽度与空间[^1]。



### 二. 两个个例子

* 第一个例子来自于 *Stanford CS231n Spring 2018*  的 *assignment1*[^2]。***这个例子总结了一般性向量化编程的思想与方法。***

> 在此例子中我们要实现利用 *l2 distance* 的 *KNN* 算法，输入 *X* 是一个 *num_test \* D* 的矩阵，*self.X_train* 是一个 *num_train \* D* 的矩阵，由这两个矩阵计算 *dists* 一个 *num_test \* num_train* 的矩阵记录了所有测试图片和训练图片的L2距离。

```python
def compute_distances_no_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using no explicit loops.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train)) 
    #########################################################################
    # TODO:                                                                 #
    # Compute the l2 distance between all test points and all training      #
    # points without using any explicit loops, and store the result in      #
    # dists.                                                                #
    #                                                                       #
    # You should implement this function using only basic array operations; #
    # in particular you should not use functions from scipy.                #
    #                                                                       #
    # HINT: Try to formulate the l2 distance using matrix multiplication    #
    #       and two broadcast sums.                                         #
    #########################################################################
    
    #########################################################################
    #                         END OF YOUR CODE                              #
    #########################################################################
    return dists
```

针对这个问题我们可以提出4种实现方式：

 1. **全循环计算**：两层循环，先对测试图片进行循环，接着对训练图片进行循环

    ```python
    for i in range(num_test):
      for j in range(num_train):
        dists[i][j] = sum(np.power(abs(X[i]-self.X_train[j]),2))
    ```

 2. **部分向量化计算**：一层循环，对测试图片循环，每个测试图片对多个训练图片进行向量运算

    ```python
    for i in range(num_test):
      dists[i] = np.sum(np.power((X[i:i+1] - self.X_train), 2), axis = 1)
    ```

 3. **完全向量化**： 构造多维数组进行计算

    ```python
    dists = np.sum(np.square(X[:,:,np.newaxis] - self.X_train.T[np.newaxis,:]), axis = 1)
    ```

 4. **利用算法优化**：利用 (a-b)^2 = a^2 + b^2 - 2ab 进行运算

    ```python
    Xts2 = np.sum(np.square(X), axis = 1)
    Xtr2 = np.sum(np.square(self.X_train), axis = 1)
    Xts_Xtr  = np.dot(X, self.X_train.T)
    dists = Xts2[:,np.newaxis] + Xtr2[np.newaxis,:] - 2*Xts_Xtr
    ```



在相同环境下我们对这四种方法进行了测试，运行结果如下：

|    实现方法    | 运行时间 / S |
| :------------: | :----------: |
|   全循环计算   |    10.88     |
| 部分向量化计算 |     4.20     |
|   完全向量化   |     0.77     |
|  利用算法优化  |    0.014     |

从以上结果可以看出，向量化程度越高代码执行效率就越高。在完全向量化之后我们还可以通过利用算法进行优化，方法4相比于方法3代码量增加了但是标量乘法的个数少了很多倍所以运行速度会更快。知道了这点后我们可以思考三个问题：

1. **什么情况下我代码中的循环可以被向量化？**

   我的回答：注意循环中数据存不存在前后依赖关系。如果存在依赖那么循环的计算就有先后关系，这时候就不存在让多个循环并发执行。（这里注意sum()并不是真正意义上的向量化运算，它隐含了循环。之所以sum()函数会比手写循环快是因为循环被当做interpreted Python bytecode执行而sum()是C code执行[^3]）

2. **在知道循环可以被向量化之后，有没有统一的向量化编程方法？**

   我的回答：构造更高维的数组总是一种值得尝试的方法，从本质上来说这也是让没有数据依赖的循环变成某一维度的向量。

3. **如何进一步对代码进行优化？**

   我的回答：进行算法层面的优化使得基本运算操作数量减少（乘法、加法等等），这没有统一的方法需要对算法知识的积累与心得。

---



* 第二个例子同样来自于 *Stanford CS231n Spring 2018*  的 *assignment1*[^2]。***这个例子注重总结在机器学习中完成梯度下降时向量化求梯度的方法。***

> 在此例子中我们要实现利用 SVM算法，先看一下直观的实现方法：

```python
def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:, j] += X[i]
        dW[:, y[i]] += -X[i]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW = dW/num_train
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += reg * 2 * W
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  return loss, dW
```

> 接着我们尝试对此进行向量化计算：

```python
def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  num_classes = W.shape[1]
  num_train = X.shape[0]

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  S = np.dot(X, W)
  mask = np.reshape((list(range(num_train)) + list(y)), (2, -1)).tolist()
  S0 = S - S[mask].reshape(-1,1) + 1
  S1 = np.maximum(S0, 0)
  loss = (np.sum(S1) - num_train)/num_train
  loss += reg * np.sum(W * W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  mask_margin = np.zeros((num_train, num_classes)) # where margin[i,j]>=0 = 1
  mask_margin[S0>0] = 1
  mask_XW = np.ones((num_train, num_classes))
  mask_XW = mask_XW * mask_margin
  y_sum = np.sum(mask_margin, axis=1)
  mask_XW[np.arange(num_train), y] -= y_sum
  dW = np.dot(X.T, mask_XW)/num_train
  dW += reg * W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW

```

还没写完，回顾起来竟然还有些地方没有想清楚，明天继续，感受到时间的宝贵

### 参考文献

[^1]: Computer Organization and Design：The Hardware/Software Interface，Fifth Edition
[^2]: <https://cs231n.github.io/assignments2018/assignment1/>
[^3]: <https://stackoverflow.com/questions/24578896/python-built-in-sum-function-vs-for-loop-performance>




# Coursera DeepLearning by Andrew Ng
> handwriting notes of introductory deep learning course :  https://www.coursera.org/specializations/deep-learning



## Resources:
### Video Course
- [Coursera](https://www.coursera.org/specializations/deep-learning#courses)
- [Bilibili](https://www.bilibili.com/video/BV1FT4y1E74V/?vd_source=d0416378a50b5f05a80e1ed2ccc0792f)

### Assignment && Quiz
- [coursera-deep-learning-specialization](https://github.com/amanchadha/coursera-deep-learning-specialization)

### Others' Notes of this course
- [fengdu78](https://github.com/fengdu78/deeplearning_ai_books)
- [bighuang624](https://kyonhuang.top/Andrew-Ng-Deep-Learning-notes/#/)

### Further
- [Dive into Deep Learning-LiMu](https://d2l.ai/)

### Pytorch
- [PyTorch for Deep Learning & Machine Learning](https://github.com/mrdbourke/pytorch-deep-learning)



## Contents
### Chapter 1: Neural Networks and Deep Learning

**Week 1**: Introduction to Deep Learning

**Week 2**: Basics of Neural Network programming

2.1 Binary Classification

2.2 Logistic Regression 

2.3 Logistic Regression Cost Function

2.4 Gradient Descent

2.5 Derivatives

2.6 More Derivative Examples

2.7 Computation Graph

2.8 Derivatives with a Computation Graph  

2.9 Logistic Regression Gradient Descent  

2.10 Gradient Descent on m Examples  

2.11 Vectorization  

2.12 More Examples of Vectorization 

2.13 Vectorizing Logistic Regression  

2.14 Vectorizing Logistic Regression's Gradient 

2.15 Broadcasting in Python 

2.16 A note on python or numpy vectors 

2.17 Jupyter/iPython Notebooks快速入门 Quick tour of Jupyter/iPython Notebooks 

2.18 Explanation of logistic regression cost function 

**Week 3**： Shallow neural networks 

3.1 Neural Network Overview 

3.2 Neural Network Representation  

3.3 Computing a Neural Network's output 

3.4 Vectorizing across multiple examples 

3.5 Justification for vectorized implementation 

3.6 Activation functions  

3.7 why need a nonlinear activation function?  

3.8 Derivatives of activation functions  

3.9 Gradient descent for neural networks  

3.10 Backpropagation intuition  

3.11 Random+Initialization 

**Week 4**: Deep Neural Networks 

4.1 Deep L-layer neural network  

4.2 Forward and backward propagation  

4.3 Forward propagation in a Deep Network 

4.4 Getting your matrix dimensions right  

4.5 Why deep representations? 

4.6 Building blocks of deep neural networks 

4.7 Parameters vs Hyperparameters  

4.8 What does this have to do with the brain? 

### Chapter 2: Improving Deep Neural Networks:Hyperparameter tuning, Regularization and Optimization 

**Week 1**：Practical aspects of Deep Learning  

1.1 Train / Dev / Test sets  

1.2 Bias /Variance  

1.3 Basic Recipe for Machine Learning  

1.4 Regularization 

1.5 Why regularization reduces overfitting? 

1.6 Dropout Regularization 

1.7 dropout Understanding Dropout 

1.8 Other regularization methods 

1.9 Normalizing inputs 

1.10 Vanishing / Exploding gradients 

1.11 Weight Initialization for Deep NetworksVanishing /Exploding gradients  

1.12 Numerical approximation of gradients 

1.13 Gradient checking 

1.14 Gradient Checking Implementation Notes  

**Week2**: Optimization algorithms  

2.1 Mini-batch gradient descent  

2.2 Understanding Mini-batch gradient descent 

2.3 Exponentially weighted averages 

2.4 Understanding Exponentially weighted averages  

2.5 Bias correction in exponentially weighted averages 

2.6 Gradient descent with momentum 

2.7 RMSprop——root mean square prop RMSprop 

2.8 Adam optimization algorithm 

2.9 Learning rate decay 

2.10 The problem of local optima 

**Week3**: Hyperparameter tuning, Batch Normalization and Programming Frameworks 

3.1 Tuning process  

3.2 Using an appropriate scale to pick hyperparameters 

3.3 Hyperparameters tuning in practice: Pandas vs. Caviar 

3.4 Normalizing activations in a network  

3.5 Fitting Batch Norm into a neural network 

3.6 Why does Batch Norm work? 

3.7 Batch Norm at test time 

3.8 Softmax Regression 

3.9 Training a softmax classifier  

3.10 Deep learning frameworks  

3.11 TensorFlow  

### Chapter 3: Structuring Machine Learning Projects 

**Week 1**: ML Strategy  1  

1.1 Why ML Strategy  

1.2 Orthogonalization  

1.3 Single number evaluation metric  

1.4 Satisficing and Optimizing metric 

1.5 Train/dev/test distributions  

1.6 Size of the dev and test sets  

1.7 When to change dev/test sets and metrics  

1.8 Why human-level performance?  

1.9 Avoidable bias  

1.10 Understanding human-level performance  

1.11 Surpassing human-level performance  

1.12 Improving your model performance  

**Week 2**: ML Strategy  2  

2.1 Carrying out error analysis  

2.2 Cleaning up incorrectly labeled data  

2.3 Build your first system quickly, then iterate  

2.4 Training and testing on different distributions  

2.5 Bias and Variance with mismatched data distributions  

2.6 Addressing data mismatch  

2.7 Transfer learning  

2.8 Multi-task learning  

2.9 What is end-to-end deep learning?  

2.10 Whether to use end-to-end deep learning  

### Chapter 4: Convolutional Neural Networks 

**Week 1**: Foundations of Convolutional Neural Networks 

1.1	Computer vision 

1.2	Edge detection example 

1.3	 More edge detection 

1.4	Padding	

1.5	Strided convolutions 	

1.6	Convolutions over volumes 	

1.7	One layer of a convolutional network 	

1.8	A simple convolution network example 	

1.9	Pooling layers 	

1.10 Convolutional neural network example 

1.11 Why convolutions? 

**Week 2**: Deep convolutional models: case studies 

2.1 Why look at case studies? 

2.2 Classic networks 

2.3 Residual Networks  ResNets  

2.4 Why ResNets work? 	

2.5 Network in Network and 1×1 convolutions 

2.6 Inception network motivation 	

2.7 Inception network 	

2.8 Using open-source implementations 	

2.9 Transfer Learning 	

2.10 Data augmentation 	

2.11 The state of computer vision 	

**Week 3**: Object detection 

3.1 Object localization 

3.2 Landmark detection 

3.3 Object detection 

3.4 Convolutional implementation of sliding windows 

3.5 Bounding box predictions 

3.6 Intersection over union 

3.7 Non-max suppression 

3.8 Anchor Boxes

3.9 Putting it together: YOLO algorithm 

3.10 Region proposals  Optional  

**Week 4**: Special applications: Face recognition &Neural style transfer 

4.1 What is face recognition? 

4.2 One-shot learning 

4.3 Siamese network 

4.4 Triplet 损失 

4.5 Face verification and binary classification 

4.6 What is neural style transfer? 

4.7 What are deep ConvNets learning? 

4.8 Cost function 

4.9 Content cost function 

4.10 Style cost function 

4.11 1D and 3D generalizations of models 

### Chapter 5: Sequence Models 

**Week 1**: Recurrent Neural Networks 

1.1 Why Sequence Models? 

1.2 Notation 

1.3 Recurrent Neural Network Model 

1.4 Backpropagation through time 

1.5 Different types of RNNs 

1.6 Language model and sequence generation 

1.7 Sampling novel sequences 

1.8 Vanishing gradients with RNNs 

1.9 Gated Recurrent Unit GRU  

1.10 LSTM long short term memory unit 

1.11 Bidirectional RNN 

1.12 Deep RNNs 

**Week 2**: Natural Language Processing and Word Embeddings 

2.1 Word Representation 

2.2 Using Word Embeddings 

2.3 Properties of Word Embeddings 

2.4 Embedding Matrix 

2.5 Learning Word Embeddings 

2.6 Word2Vec

2.7 Negative Sampling 

2.8 GloVe Word Vectors 

2.9 Sentiment Classification 

2.10 Debiasing Word Embeddings 

**Week 3**: Sequence models & Attention mechanism 

3.1 Basic Models 

3.2 Picking the most likely sentence 

3.3 Beam Search 

3.4 Refinements to Beam Search 

3.5 Error analysis in beam search 

3.6  Bleu Score  optional  

3.7 Attention Model Intuition 

3.8 Attention Model 

3.9 Speech recognition 

3.10 Trigger Word Detection 





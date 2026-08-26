# CSE 815 — Machine Learning (Theory)
## OBE-Based 12-Week Lecture Plan

**Department of CSE | University of Chittagong**
**Credits:** 3 | **Contact hours:** 3 lecture hours/week × 12 weeks
**Pre-requisite:** CSE 221 — Data Structures, CSE 411 — Algorithms, MAT 403 — Probability & Statistics
**Co-requisite:** CSE 816 — Machine Learning Lab (1 credit)
**Instructor:** Rokan Uddin Faruqui, Professor, Dept. of CSE — rufaruqui@cu.ac.bd

Include descriptive and design questions at the end of each lectures. Following standard materials teaching in CS udendrad course. Include book reference just one or maximum two for reading and exam preperations.

**CSE 815: Machine Learning **

**Module 1: Linear Regression, Classification & Regularization**
  Week 1: Introduction to Machine Learning
   Welcome to machine learning! • 3 minutes
   Applications of machine learning • 4 minutes
   What is machine learning? • 5 minutes
   Supervised learning part 1 • 7 minutes
   Supervised learning part 2 • 7 minutes
   Unsupervised learning part 1 • 
   Linear regression model part 2 • 7 minutes
   Cost function formula • 9 minutes
   Cost function intuition • 16 minutes
   Visualizing the cost function • 9 minutes
   Visualization examples • 6 minutes
   Gradient descent • 8 minutes
   Implementing gradient descent • 10 minutes
   Gradient descent intuition • 7 minutes
   Learning rate • 9 minutes
   Gradient descent for linear regression • 7 minutes
   Running gradient descent • 6 minutes

  Week 2: Regression with Multiple input variables
   This week, you'll extend linear regression to handle multiple input features. You'll also learn some methods for improving your model's training and performance, such as vectorization, feature scaling, feature engineering and polynomial regression. At the end of the week, you'll get to practice implementing linear regression in code.

    Multiple features•10 minutes
    Vectorization part 1•7 minutes
    Vectorization part 2•7 minutes
    Gradient descent for multiple linear regression•8 minutes
    Feature scaling part 1•7 minutes
    Feature scaling part 2•8 minutes
    Checking gradient descent for convergence•6 minutes
    Choosing the learning rate•6 minutes
    Feature engineering•3 minutes
    Polynomial regression•6 minutes
   
  Week3: Classifications
    This week, you'll learn the other type of supervised learning, classification. You'll learn how to predict categories using the logistic regression model. You'll learn about the problem of overfitting, and how to handle this problem with a method called regularization. You'll get to practice implementing logistic regression with regularization at the end of this week!

    Motivations•10 minutes
    Logistic regression•10 minutes
    Decision boundary•11 minutes
    Cost function for logistic regression•12 minutes
    Simplified Cost Function for Logistic Regression•6 minutes
    Gradient Descent Implementation•7 minutes
    The problem of overfitting•12 minutes
    Addressing overfitting•8 minutes
    Cost function with regularization•9 minutes
    Regularized linear regression•9 minutes
    Regularized logistic regression•6 minutes

**Module 2: Neural Networks, ML Advice & Decision Trees** 
   Week1: Neural Netorks
     This week, you'll learn about neural networks and how to use them for classification tasks. You'll use the TensorFlow framework to build a neural network with just a few lines of code. Then, dive deeper by learning how to code up your own neural network in Python, "from scratch". Optionally, you can learn more about how neural network computations are implemented efficiently using parallel processing (vectorization).
     Neurons and the brain•11 minutes
    Demand Prediction•16 minutes
    Example: Recognizing Images•7 minutes
    Neural network layer•10 minutes
    More complex neural networks•8 minutes
    Inference: making predictions (forward propagation)•5 minutes
    Inference in Code•7 minutes
    Data in TensorFlow•11 minutes
    Building a neural network•8 minutes
    Forward prop in a single layer•5 minutes
    General implementation of forward propagation•8 minutes
    Is there a path to AGI?•11 minutes
    How neural networks are implemented efficiently•4 minutes
    Matrix multiplication•9 minutes
    Matrix multiplication rules•10 minutes
    Matrix multiplication code•6 minutes

  Week2: Neural Network Training
   This week, you'll learn how to train your model in TensorFlow, and also learn about other important activation functions (besides the sigmoid function), and where to use each type in a neural network. You'll also learn how to go beyond binary classification to multiclass classification (3 or more categories). Multiclass classification will introduce you to a new activation function and a new loss function. Optionally, you can also learn about the difference between multiclass classification and multi-label classification. You'll learn about the Adam optimizer, and why it's an improvement upon regular gradient descent for neural network training. Finally, you will get a brief introduction to other layer types besides the one you've seen thus far.
   TensorFlow implementation•4 minutes
    Training Details•13 minutes
    Alternatives to the sigmoid activation•5 minutes
    Choosing activation functions•8 minutes
    Why do we need activation functions?•6 minutes
    Multiclass•3 minutes
    Softmax•12 minutes
    Neural Network with Softmax output•7 minutes
    Improved implementation of softmax•9 minutes
    Classification with multiple outputs (Optional)•4 minutes
    Advanced Optimization•6 minutes
    Additional Layer Types•9 minutes
    What is a derivative? (Optional)•23 minutes
    Computation graph (Optional)•19 minutes
    Larger neural network example (Optional)•10 minutes
  Week3: 
   This week you'll learn best practices for training and evaluating your learning algorithms to improve performance. This will cover a wide range of useful advice about the machine learning lifecycle, tuning your model, and also improving your training data.

   Deciding what to try next•4 minutes
    Evaluating a model•10 minutes
    Model selection and training/cross validation/test sets•14 minutes
    Diagnosing bias and variance•11 minutes
    Regularization and bias/variance•10 minutes
    Establishing a baseline level of performance•9 minutes
    Learning curves•12 minutes
    Deciding what to try next revisited•9 minutes
    Bias/variance and neural networks•11 minutes
    Iterative loop of ML development•8 minutes
    Error analysis•8 minutes
    Adding data•14 minutes
    Transfer learning: using data from a different task•12 minutes
    Full cycle of a machine learning project•9 minutes
    Fairness, bias, and ethics•10 minutes
    Error metrics for skewed datasets•12 minutes
    Trading off precision and recall

**Module 3: Unsupervised Learning, Recommenders & Reinforcement Learning** 
 Week1:
   This week, you will learn two key unsupervised learning algorithms: clustering and anomaly detection
   What is clustering?•4 minutes
    K-means intuition•7 minutes
    K-means algorithm•10 minutes
    Optimization objective•11 minutes
    Initializing K-means•9 minutes
    Choosing the number of clusters•7 minutes
    Finding unusual events•12 minutes
    Gaussian (normal) distribution•11 minutes
    Anomaly detection algorithm•12 minutes
    Developing and evaluating an anomaly detection system•12 minutes
    Anomaly detection vs. supervised learning•8 minutes
    Choosing what features to use•15 minutes

  Week2: Recommender Systems
    Making recommendations•6 minutes
    Using per-item features•11 minutes
    Collaborative filtering algorithm•14 minutes
    Binary labels: favs, likes and clicks•8 minutes
    Mean normalization•9 minutes
    TensorFlow implementation of collaborative filtering•12 minutes
    Finding related items•7 minutes
    Collaborative filtering vs Content-based filtering•10 minutes
    Deep learning for content-based filtering•10 minutes
    Recommending from a large catalogue•8 minutes
    Ethical use of recommender systems•11 minutes
    TensorFlow implementation of content-based filtering•5 minutes
    Reducing the number of features (optional)•12 minutes
    PCA algorithm (optional)•18 minutes
    PCA in code (optional)•11 minutes
  Week3: Reinforcement Learning
  This week, you will learn about reinforcement learning, and build a deep Q-learning neural network in order to land a virtual lunar lander on Mars!
  What is Reinforcement Learning?•9 minutes
    Mars rover example•7 minutes
    The Return in reinforcement learning•10 minutes
    Making decisions: Policies in reinforcement learning•3 minutes
    Review of key concepts•6 minutes
    State-action value function definition•11 minutes
    State-action value function example•5 minutes
    Bellman Equation•13 minutes
    Random (stochastic) environment (Optional)•8 minutes
    Example of continuous state space applications•6 minutes
    Lunar lander•6 minutes
    Learning the state-value function•17 minutes
    Algorithm refinement: Improved neural network architecture•3 minutes
    Algorithm refinement: ϵ-greedy policy•9 minutes
    Algorithm refinement: Mini-batch and soft updates (optional)•12 minutes
    
---
**CSE 816: Machine Learning Lab (Practical & Implementation)**
**Module 1: Regression and Classification Labs**
Week 1: 
    Lab Setup: Python and Jupyter Notebooks • 60 minutes
    Model Representation • 60 minutes
    Cost Function Implementation • 60 minutes
    Gradient Descent • 60 minutes
Week 2:
    Practice lab: Linear regression•180 minutes 
    Optional lab: Python, NumPy and vectorization•60 minutes
    Optional Lab: Multiple linear regression•60 minutes
    Optional Lab: Feature scaling and learning rate•60 minutes
    Optional lab: Feature engineering and Polynomial regression•60 minutes
    Optional lab: Linear regression with scikit-learn•60 minutes 
Week 3: 
    Practice lab: logistic regression•180 minutes 
    Optional lab: Classification•60 minutes
    Optional lab: Sigmoid function and logistic regression•60 minutes
    Optional lab: Decision boundary•60 minutes
    Optional lab: Logistic loss•60 minutes
    Optional lab: Cost function for logistic regression•60 minutes
    Optional lab: Gradient descent for logistic regression•60 minutes
    Optional lab: Logistic regression with scikit-learn•60 minutes
    Optional lab: Overfitting•60 minutes
    Optional lab: Regularization•60 minutes

**Module 2: Neural Networks, ML Advice & Decision Trees** 
Week 1:
    Practice Lab: Neural Networks for Binary Classification•180 minutes 
    Neurons and Layers•10 minutes
    Coffee Roasting in Tensorflow•10 minutes
    CoffeeRoastingNumPy•60 minutes
 Week 2:
   Practice Lab: Neural Networks for Multiclass classification •180 minutes 
   ReLU activation•60 minutes
   Softmax•60 minutes
   Multiclass•15 minutes
   Optional Lab: Derivatives•30 minutes
   Optional Lab: Back propagation•30 minutes
Week 3:
  Optional Lab: Model Evaluation and Selection•30 minutes
  Optional Lab: Diagnosing Bias and Variance•30 minutes

**Module 3: Unsupervised Learning, Recommenders & Reinforcement Learning** 
  Week 1:
    k-means•180 minutes
   Anomaly Detection•180 minutes
  Week 2:
    Collaborative Filtering Recommender Systems•180 minutes
    Deep Learning for Content-Based Filtering•180 minutes 
    PCA and data visualization (optional)•30 minutes
   Week 3:
    Reinforcement Learning•180 minutes 
S   tate-action value function (optional lab)•60 minutes




Quizes:
   Practice quiz: Supervised vs unsupervised learning • 15 minutes
   Practice quiz: Regression • 10 minutes
   Practice quiz: Train the model with gradient descent • 10 m
   Practice quiz: Multiple linear regression•15 minutes
   Practice quiz: Gradient descent in practice•30 minutes
   Practice quiz: Classification with logistic regression•30 minutes
   Practice quiz: Cost function for logistic regression•30 minutes
   Practice quiz: Gradient descent for logistic regression•30 minutes
   Practice quiz: The problem of overfitting•30 minutes
   Practice quiz: Neural networks intuition•10 minutes
   Practice quiz: Neural network model•10 minutes
   Practice quiz: TensorFlow implementation•10 minutes
   Practice quiz: Neural network implementation in Python•10 minutes
   Practice quiz: Neural Network Training•30 minutes
   Practice quiz: Activation Functions•30 minutes
   Practice quiz: Multiclass Classification•30 minutes
   Practice quiz: Additional Neural Network Concepts•30 minutes
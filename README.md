# MachineLearning
My machine learning projects and models
## AI Flashcards ##
## AI Project ##
# RNN Model 
This Python script implements a comprehensive machine learning pipeline to predict NBA playoff rankings. It uses a hybrid neural network model combining a Recurrent Neural Network (RNN, specifically LSTM) for processing sequences of individual player statistics and a Multi-Layer Perceptron (MLP) for aggregated team-level features.

The overall workflow is:
Data Loading: Ingests player statistics and actual playoff team ranks from multiple Excel files, organized by season.
Data Preprocessing: Cleans, standardizes team names, aggregates player stats to a team-season level, creates sequences of player data for RNN input, merges this with playoff ranks, and scales all features.
Model Definition: Defines a custom PyTorch Dataset (NBADataset) for handling the specific data structure and the RNNNBAModel architecture.
Training and Evaluation: Iterates through NBA seasons. For each season, it splits data into training and testing sets, trains a fresh instance of the RNNNBAModel, evaluates its performance on all data for that season, and converts the model's raw output scores into predicted ranks.
Results Aggregation: Combines the prediction results from all seasons and saves them into a single CSV file.
The script is designed to be end-to-end, from raw data ingestion to final ranked predictions, with robust error handling and data validation steps throughout.


# BN
This Jupyter Notebook uses the pgmpy library to construct and visualize a Bayesian Network (BN). The network models a hypothetical scenario related to credit card transaction fraud. It defines several discrete random variables (nodes) representing factors like whether the cardholder is traveling, owns a computer, or if a transaction is fraudulent, foreign, or internet-based. The script specifies the causal relationships (arcs) between these variables and quantifies these relationships using Conditional Probability Distributions (CPDs). Finally, it validates the constructed model and visualizes its structure. The overall purpose is to create a probabilistic graphical model that can be used for reasoning about the likelihood of fraud given certain observed conditions.

# CNN
This Jupyter Notebook implements a Convolutional Neural Network (CNN) using PyTorch to perform image classification on the CIFAR-10 dataset. The primary goal is to train and evaluate this CNN architecture under various hyperparameter settings (number of neurons in a fully connected layer, number of training epochs, and type of optimizer) to observe their impact on the model's test accuracy. Finally, it summarizes these results in a table for comparison and provides a general explanation of how to interpret them.

# EvaluateOnNetwork
This Jupyter Notebook provides a step-by-step guide to building, training, evaluating, and utilizing a simple feedforward neural network (Multi-Layer Perceptron - MLP) for a multi-class classification task. It uses the well-known Iris dataset as an example. The notebook demonstrates a complete machine learning workflow with PyTorch, including:

Defining the neural network architecture.
Loading and preprocessing data (the Iris dataset).
Splitting data into training and testing sets.
Training the network using an optimizer and a loss function.
Visualizing the training process (loss over epochs).
Evaluating the trained model's performance on unseen test data.
Making predictions on new, individual data points.
Saving the trained model's parameters and loading them into a new model instance for future use.
The overall goal is to illustrate the fundamental concepts of creating and working with neural networks in PyTorch for a classification problem.

# SVM
his Jupyter Notebook is designed to perform text classification on a dataset represented by word occurrences. It aims to:

Load text data (document-word pairs and corresponding labels) and a vocabulary list.
Train and evaluate a Gaussian Naïve Bayes classifier.
For the Naïve Bayes model, identify and list the top 10 most "discriminative" words (features) based on a calculated score reflecting how differently each word's presence is modeled between the two classes.
Train and evaluate Support Vector Machine (SVM) classifiers with both linear and polynomial kernels.
Discuss the conceptual approach to visualizing decision boundaries for high-dimensional text data, as direct plotting is not feasible.
The overall goal is to apply and compare these different classification algorithms on the text data and understand feature importance in the context of Naïve Bayes. The notebook anticipates potential issues with file loading and variable definitions, as suggested by the error tracebacks in the original ipynb file.

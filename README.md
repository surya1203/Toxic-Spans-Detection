## Toxic Spans Detection

This project is the final project of Surya Giri for the course 2022F CS 584 A. The purpose of the project is to build a model for the task of Toxic Spans Detection. 

### 1. Introduction
The task of sequence labeling is a significant area of research in computational linguistics. In sequence labeling, each token (word) is assigned to a class. In this project, we aim to predict which part of a sentence or an entire sequence is toxic, instead of identifying just the toxic words. This helps us detect negative sentences even if it contains not many negative words. To accomplish this task, we use the Electra model to generate pre-trained word embeddings which saves time in pre-processing and generates better Name Entity Recognition and thus better predictions. 

### 2. Problem Formulation
The problem statement for this project is to given a sentence, extract a list of indexes that point to toxic phrases in this sentence. To make this prediction, we train the model on the word embeddings generated from the training data. Tokenization is done using the Electra transformer from Tensorflow.

Conditional Random Fields (CRF) are used to model the Conditional Distribution for the sequence labeling task. F1 score is used as the evaluation metric for the set.

### 3. Method
The dataset used in the project is divided into two sets: training data and testing data. The training data is further split to generate a validation dataset. 

The Electra model is used to generate pre-trained word embeddings, and the CRF model is used to perform sequence labeling to identify the toxic spans. The F1 score is used to evaluate the performance of the model.

### Files in the Repository
1. `README.md`: Provides an overview of the project.

2. `report.pdf`: Contains the complete report of the project.

3. `src/`: Contains the source code for the project.

4. `src/data/`: Contains the data for the project.

5. `src/models/`: Contains the trained models for the project.

### How to Run the Code
1. Clone the repository to your local machine.

2. Install the necessary dependencies using the `requirements.txt` file.

3. Navigate to the `src` directory.

4. Run the `train.py` file to train the model.

5. Run the `test.py` file to evaluate the performance of the model.

### Conclusion
In conclusion, this project presents an effective approach to detect toxic spans in a sentence. The Electra model for generating pre-trained word embeddings and the CRF model for sequence labeling are used to achieve this task. The F1 score is used to evaluate the performance of the model.

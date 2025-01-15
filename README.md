Cancer Prediction using Machine Learning
This project involves building a machine learning model to predict whether cancer will recur after treatment. The goal is to use various features such as age, gender, tumor size, family history, and treatment response to predict if cancer will come back (recurrence) after cure.

Table of Contents:
Project Description
Dataset
Preprocessing
Feature Selection
Modeling
Evaluation Metrics
Dependencies

Dataset:
The dataset used in this project includes the following variables:
[Gender,Race,SmokingStatus,FamilyHistory,CancerType,Stage,TreatmentType,TreatmentResponse,Recurrence (target variable),GeneticMarker,HospitalRegion,Age,BMI,TumorSize,SurvivalMonths]

Preprocessing:
Categorical variables are encoded using One-Hot Encoding.
The Recurrence variable is encoded using Label Encoding.
Numeric features are normalized to ensure that all variables contribute equally to the model performance.
Feature Selection
We use Recursive Feature Elimination (RFE) to select the most important features and reduce the complexity of the model. The top 10 features are selected for training.

Modeling:
Three classification models are trained and evaluated:

1.Logistic Regression
2.Random Forest Classifier
3.Support Vector Machines (SVM)
The models are trained on the top 10 selected features and evaluated using accuracy, precision, recall, and F1-score.

Evaluation Metrics:
The following metrics are used to evaluate model performance:

Accuracy: The percentage of correct predictions.
Precision: The proportion of positive predictions that were actually correct.
Recall: The proportion of actual positives that were correctly identified.

Dependencies:
This project requires the following Python libraries:

pandas
numpy
scikit-learn
matplotlib
seaborn

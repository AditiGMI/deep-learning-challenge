# deep-learning-challenge
Neural Networks 

Overview of the analysis:
From Alphabet Soup’s business team, we have received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as:
EIN and NAME—Identification columns
APPLICATION_TYPE—Alphabet Soup application type
AFFILIATION—Affiliated sector of industry
CLASSIFICATION—Government organization classification
USE_CASE—Use case for funding
ORGANIZATION—Organization type
STATUS—Active status
INCOME_AMT—Income classification
SPECIAL_CONSIDERATIONS—Special considerations for application
ASK_AMT—Funding amount requested
IS_SUCCESSFUL—Was the money used effectively

The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. With our knowledge of machine learning and neural networks, we'll use the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.


Step 1: Preprocess the Data

Using Pandas and scikit-learn’s StandardScaler(), we’ll need to preprocess the dataset. This step prepares for Step 2, where we'll compile, train, and evaluate the neural network model.

1. Read in the charity_data.csv to a Pandas DataFrame, and be sure to identify the following in your dataset:
The variable that are the target for the model : 'APPLICATION_TYPE'
The variable that are the feature for the model : 'CLASSIFICATION'
2. Drop the EIN and NAME columns.
3. Determine the number of unique values for each column.
4. For columns that have more than 10 unique values, determine the number of data points for each unique value.
5. Use the number of data points for each unique value to pick a cutoff point to bin "rare" categorical variables together in a new value, Other, and then check if the binning was successful.
6. Use pd.get_dummies() to encode categorical variables.
7. Split the preprocessed data into a features array, X, and a target array, y. Use these arrays and the train_test_split function to split the data into training and testing datasets.
8. Scale the training and testing features datasets by creating a StandardScaler instance, fitting it to the training data, then using the transform function.


Step 2: Compile, Train, and Evaluate the Model

Using TensorFlow, we’ll design a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup-funded organization will be successful based on the features in the dataset. We’ll need to think about how many inputs there are before determining the number of neurons and layers in your model. Once we’ve completed that step, we’ll compile, train, and evaluate your binary classification model to calculate the model’s loss and accuracy.
1. We will use the file in Google Colab in which you performed the preprocessing steps from Step 1.
Create a neural network model by assigning the number of input features and nodes for each layer using TensorFlow and Keras.
2. Create the first hidden layer and choose an appropriate activation function.
3. Add a second hidden layer with an appropriate activation function.
4. Create an output layer with an appropriate activation function.
5. Check the structure of the model.
6. Compile and train the model.
7. Create a callback that saves the model's weights every five epochs.
8. Evaluate the model using the test data to determine the loss and accuracy.
9. Save and export your results to an HDF5 file. Name the file AlphabetSoupCharity.h5.

Summary: Results :
268/268 - 0s - loss: 0.5521 - accuracy: 0.7329 - 486ms/epoch - 2ms/step
Loss: 0.5520747900009155, Accuracy: 0.7329446077346802

Accuracy is 73%

My recommentdation is to run the model a few times to get better accuracy & runtime.
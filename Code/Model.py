import pandas
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras import callbacks
from sklearn.metrics import accuracy_score, auc, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
import random

# Import matplotlib pyplot safely
try:
    import matplotlib.pyplot as plt
except ImportError:
    import matplotlib

    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
from os.path import exists
import datetime


def standardize(data):
    """
    This standardizes the data into the MinMaxReduced version used for model creation
    """
    columns = data.columns.values[0:len(data.columns.values)]
    # Create the Scaler object
    scaler = preprocessing.MinMaxScaler()
    # Fit your data on the scaler object
    dataScaled = scaler.fit_transform(dataset)
    dataScaled = pandas.DataFrame(dataScaled, columns=columns)
    return dataScaled


def fitting_loops(X, Y, dataset, folder, modelname):
    ##2. Defining a Neural Network
    # Model creation
    model = Sequential()

    # Input
    # X.shape[1] is the number of columns inside of X
    # Done to remove need to alter input values every time we alter variables used (simplicity)
    model.add(Dense(X.shape[1], input_dim=X.shape[1], activation='sigmoid'))

    # Hidden Layers
    # Use for standard sized variable set
    model.add(Dense(X.shape[1] - 10, activation='sigmoid'))
    model.add(Dense(X.shape[1] - 15, activation='sigmoid'))
    model.add(Dense(X.shape[1] - 20, activation='sigmoid'))
    model.add(Dense(X.shape[1] - 30, activation='sigmoid'))

    # Output
    model.add(Dense(1, activation='sigmoid'))

    ##3. Compiling a model.
    model.compile(loss='mse', optimizer='nadam', metrics=['accuracy'])
    print(model.summary())

    # File path to hold results of learning cycle
    file = folder + str(datetime.date.today()) + "AverageHolder.csv"

    # Training Cycles
    # Each cycle's output is the next cycle's input, so the model learns for each new cycle
    for i in range(0, 50):

        ##Shuffling
        dataset = shuffle(dataset)
        ##Creating X and Y. Accident is the first column, therefore it is 0.
        X = dataset.iloc[:, 1:(len(dataset.columns) + 1)].values  # Our independent variables
        Y = dataset.iloc[:, 0].values  # Our dependent variable

        ##Splitting data into train and test.
        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.30, random_state=42)

        # If the model already exists, import and update/use it. If not, create it.
        if exists(folder + modelname):
            model.load_weights(folder + modelname)
            print("Loading Grid Model")

        # If the average holder file exists, import it. If not, create it.
        if exists(file):
            avg_holder = pandas.read_csv(file,
                                         usecols=["Train_Acc", "Train_Loss", "Test_Acc", "Test_Loss", "AUC", "TN", "FP",
                                                  "FN", "TP", "Accuracy", "Precision", "Recall", "Specificity", "FPR"])
            j = avg_holder.shape[0]

        else:
            avg_holder = pandas.DataFrame(
                columns=["Train_Acc", "Train_Loss", "Test_Acc", "Test_Loss", "AUC", "TN", "FP", "FN", "TP", "Accuracy",
                         "Precision", "Recall", "Specificity", "FPR"])
            j = avg_holder.shape[0]

        print("Cycle: ", i)

        # If the model doesn't improve over the past X epochs, exit training
        patience = 30
        stopper = callbacks.EarlyStopping(monitor='accuracy', patience=patience)
        hist = model.fit(X_train, y_train, epochs=1000, batch_size=5000, validation_data=(X_test, y_test), verbose=1,
                         callbacks=[stopper])

        # Save the weights for next run.
        model.save_weights(folder + modelname)
        print("Saved grid model to disk")

        # This is evaluating the model, and printing the results of the epochs.
        scores = model.evaluate(X_train, y_train, batch_size=5000)
        print("\nModel Training Accuracy:", scores[1] * 100)
        print("Model Training Loss:", sum(hist.history['loss']) / len(hist.history['loss']))

        # Okay, now let's calculate predictions probability.
        predictions = model.predict(X_test)

        # Then, let's round to either 0 or 1, since we have only two options.
        predictions_round = [abs(round(x[0])) for x in predictions]

        # Finding accuracy score of the predictions versus the actual Y.
        accscore1 = accuracy_score(y_test, predictions_round)
        # Printing it as a whole number instead of a percent of 1. (Just easier for me to read)
        print("Rounded Test Accuracy:", accscore1 * 100)
        # Find the Testing loss as well:
        print("Test Loss", sum(hist.history['val_loss']) / len(hist.history['val_loss']))

        ##Finding the AUC for the cycle:
        fpr, tpr, _ = roc_curve(y_test, predictions_round)
        # try:
        roc_auc = auc(fpr, tpr)
        print('AUC: %f' % roc_auc)
        # except:
        #     print("ROC Error, FPR: ", tpr, "TPR: ", tpr)

        ##Confusion Matrix:
        tn, fp, fn, tp = confusion_matrix(y_test, predictions_round).ravel()
        print(tn, fp, fn, tp)

        ##Adding the scores to the average holder file.
        avg_holder.at[j, 'Train_Acc'] = scores[1] * 100
        avg_holder.at[j, 'Train_Loss'] = sum(hist.history['loss']) / len(hist.history['loss'])
        avg_holder.at[j, 'Test_Acc'] = accscore1 * 100
        avg_holder.at[j, 'Test_Loss'] = sum(hist.history['val_loss']) / len(hist.history['val_loss'])
        try:
            avg_holder.at[j, 'AUC'] = roc_auc
        except:
            avg_holder.at[j, 'AUC'] = 'error in the matrix'
        avg_holder.at[j, 'TP'] = tp
        avg_holder.at[j, 'TN'] = tn
        avg_holder.at[j, 'FP'] = fp
        avg_holder.at[j, 'FN'] = fn

        try:
            accuracy = (tn + tp) / (tp + tn + fp + fn)
        except:
            accuracy = 0
        try:
            precision = tp / (fp + tp)
        except:
            precision = 0
        try:
            recall = tp / (fn + tp)
        except:
            recall = 0
        try:
            fprate = fp / (tn + fp)
        except:
            fprate = 0
        try:
            specificity = tn / (tn + fp)
        except:
            specificity = 0
        print(accuracy, precision, recall, specificity, fprate)
        avg_holder.at[j, 'Accuracy'] = accuracy
        avg_holder.at[j, 'Precision'] = precision
        avg_holder.at[j, 'Recall'] = recall
        avg_holder.at[j, 'Specificity'] = specificity
        avg_holder.at[j, 'FPR'] = fprate

        # Save the average holder file:
        avg_holder.to_csv(file, sep=",")

        # Saving every ten rounds,and the end, make some graphs:
        # if i % 10 == 0 or i == 49:
        #     generate_results(y_test, predictions, hist, fpr, tpr, roc_auc, i, folder)


def modelRun(X, Y, folder, modelname, testID):
    ##2. Defining a Neural Network
    # Model creation
    model = Sequential()

    # Input
    # X.shape[1] is the number of columns inside of X
    # Done to remove need to alter input values every time we alter variables used (simplicity)
    model.add(Dense(X.shape[1], input_dim=X.shape[1], activation='sigmoid'))

    # Hidden Layers
    # Use for standard sized variable set
    model.add(Dense(X.shape[1] - 10, activation='sigmoid'))
    model.add(Dense(X.shape[1] - 15, activation='sigmoid'))
    model.add(Dense(X.shape[1] - 20, activation='sigmoid'))
    model.add(Dense(X.shape[1] - 30, activation='sigmoid'))

    # Output
    model.add(Dense(1, activation='sigmoid'))

    ##3. Compiling a model.
    model.compile(loss='mse', optimizer='nadam', metrics=['accuracy'])
    print(model.summary())

    # File path to hold results of learning cycle
    file = folder + testID + " Modelling Results.csv"

    for i in range(0, 5):
        ##Splitting data into train and test.
        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.30, random_state=42)

        # If the model already exists, import and update/use it. If not, create it.
        if exists(folder + modelname):
            model.load_weights(folder + modelname)
            print("Loading Grid Model")

        # If the average holder file exists, import it. If not, create it.
        if exists(file):
            avg_holder = pandas.read_csv(file,
                                         usecols=["Train_Acc", "Train_Loss", "Test_Acc", "Test_Loss", "AUC", "TN", "FP",
                                                  "FN", "TP", "Accuracy", "Precision", "Recall", "Specificity", "FPR"])
            j = avg_holder.shape[0]

        else:
            avg_holder = pandas.DataFrame(
                columns=["Train_Acc", "Train_Loss", "Test_Acc", "Test_Loss", "AUC", "TN", "FP", "FN", "TP", "Accuracy",
                         "Precision", "Recall", "Specificity", "FPR"])
            j = avg_holder.shape[0]

        # If the model doesn't improve over the past X epochs, exit training
        patience = 30
        stopper = callbacks.EarlyStopping(monitor='accuracy', patience=patience)
        hist = model.fit(X_train, y_train, epochs=1000, batch_size=5000, validation_data=(X_test, y_test), verbose=1,
                         callbacks=[stopper])

        # Save the weights
        model.save_weights(folder + modelname)
        print("Saved grid model to disk")

        # This is evaluating the model, and printing the results of the epochs.
        scores = model.evaluate(X_train, y_train, batch_size=5000)
        print("\nModel Training Accuracy:", scores[1] * 100)
        print("Model Training Loss:", sum(hist.history['loss']) / len(hist.history['loss']))

        # Okay, now let's calculate predictions probability.
        predictions = model.predict(X_test)

        # Then, let's round to either 0 or 1, since we have only two options.
        predictions_round = [abs(round(x[0])) for x in predictions]

        # Finding accuracy score of the predictions versus the actual Y.
        accscore1 = accuracy_score(y_test, predictions_round)
        # Printing it as a whole number instead of a percent of 1. (Just easier for me to read)
        print("Rounded Test Accuracy:", accscore1 * 100)
        # Find the Testing loss as well:
        print("Test Loss", sum(hist.history['val_loss']) / len(hist.history['val_loss']))

        ##Finding the AUC for the cycle:
        fpr, tpr, _ = roc_curve(y_test, predictions_round)
        # try:
        roc_auc = auc(fpr, tpr)
        print('AUC: %f' % roc_auc)
        # except:
        #     print("ROC Error, FPR: ", tpr, "TPR: ", tpr)

        ##Confusion Matrix:
        tn, fp, fn, tp = confusion_matrix(y_test, predictions_round).ravel()
        print(tn, fp, fn, tp)

        ##Adding the scores to the average holder file.
        avg_holder.at[j, 'Train_Acc'] = scores[1] * 100
        avg_holder.at[j, 'Train_Loss'] = sum(hist.history['loss']) / len(hist.history['loss'])
        avg_holder.at[j, 'Test_Acc'] = accscore1 * 100
        avg_holder.at[j, 'Test_Loss'] = sum(hist.history['val_loss']) / len(hist.history['val_loss'])
        try:
            avg_holder.at[j, 'AUC'] = roc_auc
        except:
            avg_holder.at[j, 'AUC'] = 'error in the matrix'
        avg_holder.at[j, 'TP'] = tp
        avg_holder.at[j, 'TN'] = tn
        avg_holder.at[j, 'FP'] = fp
        avg_holder.at[j, 'FN'] = fn

        try:
            accuracy = (tn + tp) / (tp + tn + fp + fn)
        except:
            accuracy = 0
        try:
            precision = tp / (fp + tp)
        except:
            precision = 0
        try:
            recall = tp / (fn + tp)
        except:
            recall = 0
        try:
            fprate = fp / (tn + fp)
        except:
            fprate = 0
        try:
            specificity = tn / (tn + fp)
        except:
            specificity = 0
        print(accuracy, precision, recall, specificity, fprate)
        avg_holder.at[j, 'Accuracy'] = accuracy
        avg_holder.at[j, 'Precision'] = precision
        avg_holder.at[j, 'Recall'] = recall
        avg_holder.at[j, 'Specificity'] = specificity
        avg_holder.at[j, 'FPR'] = fprate

        # Save the average holder file:
        avg_holder.to_csv(file, sep=",")


def generate_results(y_test, predictions, hist, fpr, tpr, roc_auc, i, folder):
    font = {'family': 'serif',
            'weight': 'regular',
            'size': 14}
    plt.rc('font', **font)
    fig = plt.figure()
    # print(fpr, tpr)
    # exit()
    # plt.subplot(211)
    plt.plot(fpr, tpr, label='Grid ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.yticks((0, .5, 1), (0, .5, 1))
    plt.xticks((0, .5, 1), (0, .5, 1))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic curve')
    title = folder + str(datetime.datetime.today()) + 'roc' + str(i) + '.png'
    fig.savefig(title, bbox_inches='tight')
    # plt.subplot(212)
    fig = plt.figure(figsize=(24.0, 8.0))
    plt.xticks(range(0, 100), range(0, 100), rotation=90)
    plt.yticks(range(0, 2), ['No', 'Yes', ''])
    plt.ylabel('Accident')
    plt.xlabel('Record')
    plt.grid(which='major', axis='x')
    x = range(0, 100)
    plt.axhline(y=0.5, color='gray', linestyle='-')
    plt.scatter(x=x, y=predictions[0:100], s=100, c='blue', marker='x', linewidth=2)
    plt.scatter(x=x, y=y_test[0:100], s=110,
                facecolors='none', edgecolors='r', linewidths=2)
    title = folder + str(datetime.datetime.today()) + 'pred' + str(i) + '.png'
    fig.savefig(title, bbox_inches='tight')

    font = {'family': 'serif',
            'weight': 'bold',
            'size': 14}
    plt.rc('font', **font)
    fig = plt.figure()
    a1 = fig.add_subplot(2, 1, 1)
    a1.plot(hist.history['accuracy'])
    a1.plot(hist.history['val_accuracy'])
    a1.set_ylabel('Accuracy')
    a1.set_xlabel('Epoch')
    a1.set_yticks((.5, .75, 1), (.5, .75, 1))
    a1.set_xticks((0, (len(hist.history['val_accuracy']) / 2), len(hist.history['val_accuracy'])))
    a1.legend(['Train Accuracy', 'Test Accuracy'], loc='lower right', fontsize='small')
    # plt.show()
    # fig.savefig('accuracy.png', bbox_inches='tight')
    # summarize history for loss
    # fig = plt.figure()
    a2 = fig.add_subplot(2, 1, 2)
    # fig = plt.figure()
    a2.plot(hist.history['loss'])
    a2.plot(hist.history['val_loss'])
    a2.set_ylabel('Loss')
    a2.set_xlabel('Epoch')
    a2.set_yticks((0, .25, .5), (0, .25, .5))
    a2.set_xticks((0, (len(hist.history['val_loss']) / 2), len(hist.history['val_loss'])))
    a2.legend(['Train Loss', 'Test Loss'], loc='upper right', fontsize='small')
    # plt.show()
    title = folder + str(datetime.datetime.today()) + 'lossandacc' + str(
        i) + '.png'
    fig.savefig(title, bbox_inches='tight')


# The steps of creating a neural network or deep learning model
# 1. Load Data
# 2. Defining a neural network
# 3. Compile a Keras model using an efficient numerical backend
# 4. Train a model on some data.
# 5. Evaluate that model on some data!


##1. Load Data
# Depending on the size of your dataset that you're reading in, you choose either csv or feather
# Feather files are typically any file > 800 mb
# This is done because Pycharm doesn't like CSV files above a certain size (it freezes the system)
dataset = pandas.read_csv(
    "../Data/2019 Football Player Data Alt.csv")
# dataset = dataset.drop(['ID'], axis=1)
dataset = dataset.drop(['ID', 'CLEI_PreS_or_S','AnyInj_PreS_or_S','AnyInj_PreS',"CLEI_PreS",'AnyInj_S'], axis=1)
# Select which type of test you want to do: this determines what columns are used
# dataset = test_type(dataset, 6)
# Standardize the data before modelling
dataset = standardize(dataset)

# Choose a folder for storing all of the results of the code in, including the model itself
# Note, if the folder you specify doesn't exist, you'll have to create it
# These are made for code automation later on
folder = '../Modeling Results/'
modelname = "model_SIP_CLEI_FBDataAltv2Fixed.h5"

##Shuffling

dataset = random.Random(1).shuffle(dataset)
##Creating X and Y. Accident is the first column, therefore it is 0.
X = dataset.iloc[:, 1:(len(dataset.columns) + 1)].values  # Our independent variables
Y = dataset.iloc[:, 0].values  # Our dependent variable

##Steps 2-5 are inside the fitting loops method
modelRun(X, Y, folder, modelname, "FBData Alt v2, 5 Cycles")

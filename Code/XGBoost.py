import pandas
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing


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


dataset = pandas.read_csv("../Data/High School Football Data_cleaned.csv")
dataset = dataset.drop(['ID'], axis=1)

dataset = standardize(dataset)

# Creating X and Y. Accident is the first column, therefore it is 0.
X = dataset.iloc[:, 1:(len(dataset.columns) + 1)].values  # Our independent variables
Y = dataset.iloc[:, 0].values  # Our dependent variable

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.30, random_state=7)
# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

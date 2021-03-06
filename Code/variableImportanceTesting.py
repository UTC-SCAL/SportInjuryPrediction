"""
Code file for finding the important features of the dataset
"""
import pandas
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import numpy


def PCA_testing(data):
    # First, we read in our data
    data.avgReactionTime = data.avgReactionTime.astype(float)
    data.Avg_RT_App = data.Avg_RT_App.astype(float)

    # Next, we scale the data, which we have two types to use (Standard and Normalizing)
    # Standard Scaling #
    # Separating out the features
    features = data.columns.values[1:len(data.columns.values)]
    x = data.loc[:, features].values  # Separating out the target
    y = data.loc[:, ['BodyInjury']].values  # Standardizing the features
    x = StandardScaler().fit_transform(x)

    # Now, reduce the dimensionality of the dataset
    # In this step, the labels of the variables are removed, so they basically lose their meaning
    pca = PCA(n_components=10)
    principalComponents = pca.fit_transform(x)
    principalDf = pandas.DataFrame(data=principalComponents, columns=['pc1', 'pc2', 'pc3', 'pc4', 'pc5', 'pc6', 'pc7', 'pc8',
                                                                      'pc9', 'pc10'])

    # Concatenate the principal component dataframe to our dependent variable
    finalDf = pandas.concat([principalDf, data[['BodyInjury']]], axis=1)
    print(finalDf)
    finalDf.to_csv("../")

    # Print off an explanation of the variance ratios for the different principal components
    # This tells us how much information (variance) can be attributed to each of the principal components
    print(pca.explained_variance_ratio_)
    # We can also print out the correlations that each PC has on the variables in the dataset
    pcaCorrelations = pandas.DataFrame(pca.components_, columns=features, index=['pc1', 'pc2', 'pc3', 'pc4', 'pc5', 'pc6',
                                                                                 'pc7', 'pc8', 'pc9', 'pc10'])
    print(pcaCorrelations)
    pcaCorrelations.to_csv("../")


def univariateSelection(data):
    # Drop this column since it has a negative in it, and feature selection doesn't accept negatives
    # data = data.drop(['ConflictEffect'], axis=1)

    # If we want to MinMaxReduce the data (normalize it)
    # Get the columns of the data
    # columns = data.columns.values[0:len(data.columns.values)]
    # # Create the Scaler object
    # scaler = preprocessing.MinMaxScaler()
    # # Fit your data on the scaler object
    # scaled_df = scaler.fit_transform(data)
    # data = pandas.DataFrame(scaled_df, columns=columns)

    features = data.columns.values[1:len(data.columns.values)]
    X = data.loc[:, features].values  # Separating out the target variables
    y = data.loc[:, ['AnyInj_S']].values  # dependent variable

    bestFeatures = SelectKBest(score_func=chi2, k=10)
    fit = bestFeatures.fit(X, y)
    dfScores = pandas.DataFrame(fit.scores_)
    dfColumns = pandas.DataFrame(features)

    #concat two dataframes for better visualization
    featureScores = pandas.concat([dfColumns, dfScores], axis=1)
    featureScores.columns = ['Specs','Score']  # naming the dataframe columns
    print(featureScores.nlargest(10,'Score'))  # print 10 best features


def featureSelection(data, dataSource):
    features = data.columns.values[1:len(data.columns.values)]
    X = data.loc[:, features].values  # Separating out the target variables
    # dependent variable, you'll need to change this based on what dataset you're looking at
    y = data.loc[:, ['CLEI_S']].values

    model = ExtraTreesClassifier()
    model.fit(X, y)
    print(model.feature_importances_)  # use inbuilt class feature_importances of tree based classifiers

    # plot graph of feature importances for better visualization
    feat_importances = pandas.Series(model.feature_importances_, index=features)
    feat_importances.nlargest(15).plot(kind='barh')
    plt.xlim(0, .50)
    plt.title("Feature Selection on %s Data: CLEI" % dataSource)
    plt.show()


def correlationHeatmap(data, dataSource):
    # This version allows for the creation of a heatmap that removes redundancy (takes away the top right half of the
    # heatmap for less noise)
    # Create the correlation dataframe

    # If you want an easier time comparing numbers, you can take the absolute value of the correlations
    corr = abs(data.corr())
    # corr = data.corr()

    # Drop self-correlations
    dropSelf = numpy.zeros_like(corr)
    dropSelf[numpy.triu_indices_from(dropSelf)] = True
    # Generate color map
    # colormap = sns.diverging_palette(220, 10, as_cmap=True)
    # Generate the heatmap, allowing annotations and place floats in the map
    sns.heatmap(corr, cmap='Greens', annot=False, fmt='.2f', mask=dropSelf)
    # xticks
    plt.xticks(range(len(corr.columns)), corr.columns)
    # yticks
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title("Correlation Heatmap on %s Data" % dataSource)
    plt.show()


data = pandas.read_csv("../Data/UTC Football Data_cleaned.csv")
dataSource = "UTC"
data = data.drop(['ID', 'CLEI_PreS_or_S', 'AnyInj_PreS_or_S', 'CLEI_PreS', 'AnyInj_S'], axis=1)
# PCA_testing(data)
# univariateSelection(data)
# featureSelection(data, dataSource)
correlationHeatmap(data, dataSource)

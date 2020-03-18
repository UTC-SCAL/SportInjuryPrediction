import matplotlib.pyplot as plt
import pandas
import numpy


def compareGraph(preSeason, duringSeason, variable):
    # Plot setup
    xValues = numpy.arange(len(preSeason.values))  # length of x axis
    ax1 = plt.subplot(1, 1, 1)  # subplot we'll add our data onto
    w = 0.3  # weight variable for changing bar size
    plt.xticks(xValues + w / 2, preSeason.candidate_id.values)  # player ID numbers

    # Adding the data to the graph
    plt.ylabel("Player %s" % variable)
    preS_values = ax1.bar(xValues, preSeason[variable].values, width=w, color='b', align='center')
    # The trick is to use two different axes that share the same x axis, we have used ax1.twinx() method.
    ax2 = ax1.twinx()
    durS_values = ax2.bar(xValues+w, duringSeason[variable].values, width=w, color='g', align='center')
    plt.legend([preS_values, durS_values], ['Pre Season', 'During Season'])
    plt.show()


preSeason = pandas.read_csv("../Data/Query Results Pre Season.csv")
duringSeason = pandas.read_csv("../Data/Query Results During Season.csv")
print(preSeason.columns)
compareGraph(preSeason, duringSeason, "correct_percent")

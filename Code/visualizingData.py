import matplotlib.pyplot as plt
import pandas
import numpy


def compareGraph(preSeason, duringSeason, variableGraphed, variableCut, ylimit):
    # Plot setup
    xValues = numpy.arange(len(preSeason.values))  # length of x axis
    ax1 = plt.subplot(1, 1, 1)  # subplot we'll add our data onto
    w = 0.3  # weight variable for changing bar size
    plt.xticks(xValues + w / 2, preSeason.candidate_id.values)  # player ID numbers

    # Adding the data to the graph
    plt.ylabel("Player %s" % variableGraphed)
    preS_values = ax1.bar(xValues, preSeason[variableGraphed].values, width=w, color='b', align='center')

    # The trick is to use two different axes that share the same x axis, we have used ax1.twinx() method
    ax2 = ax1.twinx()

    durS_values = ax2.bar(xValues+w, duringSeason[variableGraphed].values, width=w, color='g', align='center')
    plt.legend([preS_values, durS_values], ['Pre Season', 'During Season'])
    ax1.set_ylim([0, ylimit + 50])
    ax2.set_ylim([0, ylimit + 50])
    # Change the title of the graph based on how you cut the data
    plt.title("Pre Season vs During Season (%s)" % variableCut)
    plt.show()


variableCut = 'AnyInj_PreS'  # variable used to split the data
variableGraphed = "efficiency"  # variable you want to graph
# In the cut versions of the data below, change the value after == to determine how you want the data cut
preSeason = pandas.read_csv("../Data/Query Results Pre Season.csv")
preSeason_cut = preSeason[preSeason[variableCut] == 1]
duringSeason = pandas.read_csv("../Data/Query Results During Season.csv")
duringSeason_cut = duringSeason[duringSeason[variableCut] == 1]

print(preSeason.columns)
# exit()
# Get the maximum value possible between the two lists you want to graph
ylimit = max(max(preSeason[variableGraphed].values), max(duringSeason[variableGraphed].values))

compareGraph(preSeason_cut, duringSeason_cut, variableGraphed, variableCut, ylimit)

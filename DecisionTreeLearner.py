# from treelib import Tree, Node
import graphviz as gv
import matplotlib.pyplot as plt
import math
from DataNode import DataNode
from DataNodePriorityQueue import PriorityQueue

# This funciton calculated the delta I(e) for each feature based split dataSet
# Input: The dataSet split for if a feature appears/or not within an article
# Outout: The I(e) for given data set
def computeIndividualInformationGain(inputDataSet):
    featureArticleAtheism = 0.0
    featureArticleGraphics = 0.0

    for articleNumber in inputDataSet:
        if (articleLabelDict[articleNumber] == "1"):
            featureArticleAtheism += 1
        else:
            featureArticleGraphics += 1
    totalElements = featureArticleAtheism + featureArticleGraphics
    if(totalElements == 0):
        return 0
    else:
        fractionAtheism = featureArticleAtheism / totalElements
        fractionGraphics = featureArticleGraphics / totalElements
        if (fractionAtheism == 0):
            return -(fractionGraphics * math.log(fractionGraphics, 2))
        elif (fractionGraphics == 0):
            return -(fractionAtheism * math.log(fractionAtheism, 2))
        else:
            return -(fractionAtheism * math.log(fractionAtheism, 2) + fractionGraphics * math.log(fractionGraphics, 2))

# Calculate the Split Information Gain either on average or by weight
def computeInformationGainDelta(featureAppearsIn, featureDoesNotAppearIn, method):
    infoGainFeatureOne = computeIndividualInformationGain(featureAppearsIn)
    infoGainFeatureTwo = computeIndividualInformationGain(featureDoesNotAppearIn)
    if method == "weighted":
        featureOneTotal = float(featureAppearsIn.__len__())
        featureTwoTotal = float(featureDoesNotAppearIn.__len__())
        totalElements = featureOneTotal + featureTwoTotal
        if totalElements != 0:
            return (featureOneTotal/totalElements) * infoGainFeatureOne + (featureTwoTotal/totalElements) * infoGainFeatureTwo
        else:
            return 0.0
    else:
        return (0.5) * infoGainFeatureOne + (0.5) * infoGainFeatureTwo

# calculates teh initial point estimate, how many of each type of article there are
# Input: The dataSet containing current articleIndex values
# Output: a DataNode determining which types of article is most common
def pointEstimate(dataset):
    numAtheism, numGraphics = computeNumberOfTimesEachLabelAppears(dataset)
    if numAtheism >= numGraphics: return 1
    else: return 2

# This function splits the dataSet based on if the word appears in the article or not
# Input: A word ID
# Output:
# 1) An array of all the articles in which the word appears
# 2) An Array of all the articles in which the word does not appear
def generateFeatureBasedSplitDataSets(articleDataSet, wordID):
    featureOccuresDataset = []
    featureDoesNotOccureDataset = []
    for articleIndex in articleDataSet:
        if(array[articleIndex-1][wordID-1] == 1):
            featureOccuresDataset.append(articleIndex)
        else:
            featureDoesNotOccureDataset.append(articleIndex)
    return (featureOccuresDataset, featureDoesNotOccureDataset)



# This method calculate the number of times each label appears in given dataset
# One for Atheism
# One for Graphics
def computeNumberOfTimesEachLabelAppears(articleDataset):
    numAtheism, numGraphics = 0.0, 0.0
    for index in articleDataset:
        if articleLabelDict[index] == "1":
            numAtheism += 1
        else:
            numGraphics += 1
    return numAtheism, numGraphics


# ComputeTotalInfomationContent: can be used to calculate the total Information Content, if
# its the entire set count then
def computeTotalInfomationContent(articleDataset):
    numAtheism, numGraphics = computeNumberOfTimesEachLabelAppears(articleDataset)
    total = numAtheism + numGraphics
    if(numGraphics == 0 and numAtheism == 0):
        return 0.0
        # raise Exception("Both NumGraphics and numAtheism are 0")
    numGraphics = numGraphics / total
    numAtheism = numAtheism / total
    if(numGraphics == 0):
        return - (0 + numAtheism * math.log(numAtheism, 2))
    elif(numAtheism == 0):
        return -(numGraphics * math.log(numGraphics, 2) + 0)
    else:
        return -(numGraphics * math.log(numGraphics, 2) + numAtheism * math.log(numAtheism, 2))


# Get the single element which has a maximum information gain value
def generateMaximumInformationGainElement(exampleDataset, remainingFeatures, method):
    pq = PriorityQueue()

    informationConstant = computeTotalInfomationContent(exampleDataset)
    pointestimate = pointEstimate(exampleDataset)

    for wordID in remainingFeatures:
        featureAffirmative, featureNegative = generateFeatureBasedSplitDataSets(exampleDataset, wordID)
        splitInformationGain = computeInformationGainDelta(featureAffirmative, featureNegative, method)
        deltaInformationGain = informationConstant - splitInformationGain
        pq.push(DataNode(wordID, deltaInformationGain, [featureNegative, featureAffirmative], pointestimate), deltaInformationGain)
    return pq.pop()


# Core algorithm to generate the decision tree learner
# Input: remaining Features and the word we are filtering out
# Output: The final decision tree
def calculateRemainingFeatures(remainingFeatures, word):
    filteredRemainingFeatures = []
    for wordid in remainingFeatures:
        if wordid != word:
            filteredRemainingFeatures.append(wordid)
    return filteredRemainingFeatures


# /**************************************/

# DECISION TREE LEARNER IMPLEMENTATION

# /**************************************/

graph = gv.Digraph(format='svg')
graph2 = gv.Digraph(format='svg')
def decisionTreeLearner(completeDataSet, remainingFeatures, method):
    rootNode = generateMaximumInformationGainElement(completeDataSet, remainingFeatures, method)
    rootNode.remainingFeaturesSet = calculateRemainingFeatures(remainingFeatures, rootNode.featureName)
    print str(runTrainingDataAccuracy(rootNode, 0, method))
    print str(runTestDataAccuracyTest(rootNode, 0, method))

    rootNode.nodeIndex = 0

    heapq = PriorityQueue()
    heapq.push(rootNode, rootNode.informationGain)
    index = 0
    nodeNumber = 0
    deltaGain = -1.0

    while(nodeNumber < 100):
        topFeature = heapq.pop()
        nodeNumber += 1

        for featureValue in range(0, 2):
            splitDataSet = topFeature.dataSets[featureValue]
            newInformationGainNode = generateMaximumInformationGainElement(splitDataSet, topFeature.remainingFeaturesSet, method)
            newInformationGainNode.remainingFeaturesSet = calculateRemainingFeatures(topFeature.remainingFeaturesSet, newInformationGainNode.featureName)
            index += 1
            newInformationGainNode.nodeIndex = index

            # GRAPH GNERATIONS
            if(featureValue == 0):
                topFeature.left = newInformationGainNode
                labelEdge = "No"
            else:
                topFeature.right = newInformationGainNode
                labelEdge = "Yes"

            if (method == "weighted"):
                graph.node(str(index) + " " + wordDict[newInformationGainNode.featureName] + "\\n" + str(newInformationGainNode.informationGain) + "\\n" + str(newInformationGainNode.pointEstimate))
                graph.edge(str(topFeature.nodeIndex) + " " + wordDict[topFeature.featureName] + "\\n" + str(topFeature.informationGain) + "\\n" + str(topFeature.pointEstimate),
                       str(index) + " " + wordDict[newInformationGainNode.featureName] + "\\n" + str(newInformationGainNode.informationGain) + "\\n" + str(newInformationGainNode.pointEstimate), label=labelEdge)
            else:
                graph2.node(str(index) + " " + wordDict[newInformationGainNode.featureName] + "\\n" + str(newInformationGainNode.informationGain) + "\\n" + str(newInformationGainNode.pointEstimate))
                graph2.edge(str(topFeature.nodeIndex) + " " + wordDict[topFeature.featureName] + "\\n" + str(topFeature.informationGain) + "\\n" + str(topFeature.pointEstimate),str(index) + " " + wordDict[newInformationGainNode.featureName] + "\\n" + str(
                               newInformationGainNode.informationGain) + "\\n" + str(newInformationGainNode.pointEstimate), label=labelEdge)

            # TESTING METHOD
            print str(runTrainingDataAccuracy(rootNode, nodeNumber, method))
            print str(runTestDataAccuracyTest(rootNode, nodeNumber, method))

            heapq.push(newInformationGainNode, newInformationGainNode.informationGain)

    print str(runTrainingDataAccuracy(rootNode, nodeNumber, method))
    print str(runTestDataAccuracyTest(rootNode, nodeNumber, method))

    return rootNode

def startProgram(remainingFeatures):
    weightedDecisionTree = decisionTreeLearner(range(1, numArticles + 1), remainingFeatures, "weighted")
    printDecisionTree("weighted")
    avgDecisionTree = decisionTreeLearner(range(1, numArticles + 1), remainingFeatures, "average")
    printDecisionTree("average")
    generateDataGraphs()


def printDecisionTree(name):
    if(name == "weighted"):
       graph.render(filename='dataset/'+name)
    elif(name == "average"):
       graph2.render(filename='dataset/' + name)

def generateDataGraphs():
    plt.figure(1)
    plt.subplot(211)
    plt.plot(trainWeightedPlotPointsIndex, trainWeightedPlotPointsPercent, 'r--', testWeightedPlotPointsIndex, testWeightedPlotPointsPercent, 'b--')
    plt.title("Weighted Information Gain: Training (red) and Test (blue) Accuracy")
    plt.subplot(212)
    plt.plot(trainAveragePlotPointsIndex, trainAveragePlotPointsPercent, 'b--', testAveragePlotPointsIndex, testAveragePlotPointsPercent, 'g--')
    plt.title("Average Information Gain: Training (blue) and Test (green) Accuracy")
    plt.show()


# Set up all data
def populateTestDataValue():
    # Classification of all the documents
    with open('dataset/testLabel.txt') as articleFile:
        global testArticleLabelDict
        testArticleLabelDict = {}
        testArticlesArray = articleFile.read().splitlines()
        articleIndex = 1
        for articleLabel in testArticlesArray:
            articleLabel.rstrip("\n")
            testArticleLabelDict[articleIndex] = articleLabel
            articleIndex += 1
    articleFile.close()

    global testNumArticles
    testNumArticles = len(testArticlesArray)
    print testNumArticles

    global testArrayData
    testArrayData = [[0 for x in range(numWords)] for y in range(testNumArticles)]

    testData = []
    with open('dataset/testData.txt') as trainingData:
        testData = trainingData.readlines()

        for pair in testData:
            element = pair.split("\t")
            indexPairs = [int(index) for index in element]
            x, y = indexPairs
            testArrayData[x-1][y-1] = 1
    trainingData.close()



def populateTrainingDataValues():
    with open('dataset/words.txt') as wordFile:
        wordsArray = wordFile.read().splitlines()
        global wordDict
        wordDict = {}
        wordDict[0] = None

        remainingFeatures = []
        index = 1
        for word in wordsArray:
            word.rstrip("\n")
            wordDict[index] = word
            remainingFeatures.append(index)
            index += 1
    wordFile.close()

    global numWords
    numWords = len(wordsArray)

    # Classification of all the documents
    with open('dataset/trainLabel.txt') as articleFile:
        global articleLabelDict
        articleLabelDict = {}
        articlesArray = articleFile.read().splitlines()
        articleIndex = 1
        for articleLabel in articlesArray:
            articleLabel.rstrip("\n")
            articleLabelDict[articleIndex] = articleLabel
            articleIndex += 1
    articleFile.close()

    global numArticles
    numArticles = len(articlesArray)

    global array
    array = [[0 for x in range(numWords)] for y in range(numArticles)]

    testData = []
    with open('dataset/trainData.txt') as trainingData:
        testData = trainingData.readlines()

        for pair in testData:
            element = pair.split("\t")
            indexPairs = [int(index) for index in element]
            x, y = indexPairs
            array[x-1][y-1] = 1
    trainingData.close()
    populateTestDataValue()
    startProgram(remainingFeatures)


# /**************************************/

# TESTING METHODS

# /**************************************/

testWeightedPlotPointsIndex = []
testWeightedPlotPointsPercent = []
testAveragePlotPointsIndex = []
testAveragePlotPointsPercent = []

def runTestDataAccuracyTest(rootNode, index, method):
    #     articleLabelDict
    matchingLabelTrain = 0.0
    #   docId = 1
    for docId in range(0, testNumArticles):
        assignedPointEstimate = trainDataTest(rootNode, docId, testArrayData)
        if assignedPointEstimate == int(testArticleLabelDict[docId + 1]):
            matchingLabelTrain += 1.0
    calulatedPercentage = matchingLabelTrain / len(testArticleLabelDict) * 100
    if method == "weighted":
        testWeightedPlotPointsIndex.append(index)
        testWeightedPlotPointsPercent.append(calulatedPercentage)
    else:
        testAveragePlotPointsIndex.append(index)
        testAveragePlotPointsPercent.append(calulatedPercentage)
    return calulatedPercentage



trainWeightedPlotPointsIndex = []
trainWeightedPlotPointsPercent = []
trainAveragePlotPointsIndex = []
trainAveragePlotPointsPercent = []

def runTrainingDataAccuracy(rootNode, index, method):
    matchingLabelTrain = 0.0
    for docId in range(0, numArticles):
        assignedPointEstimate = trainDataTest(rootNode, docId, array)
        if assignedPointEstimate == int(articleLabelDict[docId + 1]):
            matchingLabelTrain += 1.0
    calulatedPercentage = matchingLabelTrain / len(articleLabelDict) * 100
    if method == "weighted":
        trainWeightedPlotPointsIndex.append(index)
        trainWeightedPlotPointsPercent.append(calulatedPercentage)
    else:
        trainAveragePlotPointsIndex.append(index)
        trainAveragePlotPointsPercent.append(calulatedPercentage)
    return calulatedPercentage


def trainDataTest(node, docId, arrayToCheck):
    if node.informationGain == 0.0:
        return node.pointEstimate
    if arrayToCheck[docId - 1][node.featureName - 1] == 1:
        if node.right != None:
            return trainDataTest(node.right, docId, arrayToCheck)
        else:
            return node.pointEstimate
    else:
        if node.left != None:
            return trainDataTest(node.left, docId, arrayToCheck)
        else:
            return node.pointEstimate



def main():
    populateTrainingDataValues()



if __name__ == "__main__": main()
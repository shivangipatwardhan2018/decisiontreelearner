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
    # print inputDataSet
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
    # print featureAppearsIn
    infoGainFeatureOne = computeIndividualInformationGain(featureAppearsIn)
    infoGainFeatureTwo = computeIndividualInformationGain(featureDoesNotAppearIn)
    # print " infoGainFeatureOne: " + str(infoGainFeatureOne)
    # print featureDoesNotAppearIn
    # print " infoGainFeatureOne: " + str(infoGainFeatureTwo)
    if method == "weighted":
        featureOneTotal = float(featureAppearsIn.__len__())
        featureTwoTotal = float(featureDoesNotAppearIn.__len__())
        totalElements = featureOneTotal + featureTwoTotal
        return (featureOneTotal/totalElements) * infoGainFeatureOne + (featureTwoTotal/totalElements) * infoGainFeatureTwo
    else:
        return (1 / 2) * infoGainFeatureOne + (1 / 2) * infoGainFeatureTwo

# calculates teh initial point estimate, how many of each type of article there are
# Input: The dataSet containing current articleIndex values
# Output: a DataNode determining which types of article is most common
def pointEstimate(dataset):
    # print "       Point Estimate :    "
    numAtheism, numGraphics = computeNumberOfTimesEachLabelAppears(dataset)
    # print "    Atheism: " + str(numAtheism)
    # print "    Graphics: " + str(numGraphics)
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
    # print featureOccuresDataset, featureDoesNotOccureDataset
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


# computeTotalInfomationContent: can be used to calculate the total Information Content, if
# its the entire set count then
def computeTotalInfomationContent(articleDataset):
    numAtheism, numGraphics = computeNumberOfTimesEachLabelAppears(articleDataset)
    total = numAtheism + numGraphics
    # print "graphics: " + str(numGraphics)
    # print "atheism: " + str(numAtheism)
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


def generateMaximumInformationGainElement(exampleDataset, remainingFeatures, method):
    pq = PriorityQueue()
    # print exampleDataset
    informationConstant = computeTotalInfomationContent(exampleDataset)
    pointestimate = pointEstimate(exampleDataset)
    # print "INFORMATION CONTENT GAIN: " + str(informationConstant)
    # print remainingFeatures
    for wordID in remainingFeatures:
        # print "AT INDEX: " + str(wordID ) + " WORD: " + wordDict[wordID]
        featureAffirmative, featureNegative = generateFeatureBasedSplitDataSets(exampleDataset, wordID)

        splitInformationGain = computeInformationGainDelta(featureAffirmative, featureNegative, method)

        deltaInformationGain = informationConstant - splitInformationGain
        # print pointestimate
        # print featureAffirmative
        # print featureNegative
        pq.push(DataNode(wordID, deltaInformationGain, [featureNegative, featureAffirmative], pointestimate), deltaInformationGain)
    return pq.pop()

# Core algorithm to generate the decision tree learner
# Input:
# Output: The final decision tree
def calculateRemainingFeatures(remainingFeatures, word):
    filteredRemainingFeatures = []
    for wordid in remainingFeatures:
        if wordid != word:
            filteredRemainingFeatures.append(wordid)
    return filteredRemainingFeatures

graph = gv.Digraph(format='svg' )

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
        if assignedPointEstimate == int(testArticleLabelDict[docId+1]):
            matchingLabelTrain += 1.0
    calulatedPercentage = matchingLabelTrain/len(testArticleLabelDict) * 100
    if method == "weighted":
        testWeightedPlotPointsIndex.append(index)
        testWeightedPlotPointsPercent.append(calulatedPercentage)
    else:
        testAveragePlotPointsIndex.append(index)
        testAveragePlotPointsPercent.append(calulatedPercentage)
    # print plotPoints
    return calulatedPercentage


trainWeightedPlotPointsIndex = []
trainWeightedPlotPointsPercent = []
trainAveragePlotPointsIndex = []
trainAveragePlotPointsPercent = []
def runTrainingDataAccuracy(rootNode, index, method):
#     articleLabelDict
    matchingLabelTrain = 0.0
#   docId = 1
    for docId in range(0, numArticles):
        assignedPointEstimate = trainDataTest(rootNode, docId, array)
        if assignedPointEstimate == int(articleLabelDict[docId+1]):
            matchingLabelTrain += 1.0
    # print "CORRECT LABEL: " + str(matchingLabel)
    calulatedPercentage = matchingLabelTrain/len(articleLabelDict) * 100
    if method == "weighted":
        trainWeightedPlotPointsIndex.append(index)
        trainWeightedPlotPointsPercent.append(calulatedPercentage)
    else:
        trainAveragePlotPointsIndex.append(index)
        trainAveragePlotPointsPercent.append(calulatedPercentage)
    # print plotPoints
    return calulatedPercentage

def trainDataTest(node, docId, arrayToCheck):
    # print "******* INSIDE TEST DATA FUNCTION ******* " + str(docId) + " " + str(node.nodeIndex)

    if node.informationGain == 0.0:
        # print "Zero: " + str(node.pointEstimate)
        return node.pointEstimate
    if arrayToCheck[docId - 1][node.featureName - 1] == 1:
        # print "Checking Feature: " + wordDict[node.featureName]
        if node.right != None:
            # print "     Going Right: " + str(node.pointEstimate)
            return trainDataTest(node.right, docId, arrayToCheck)
        else:
            # print " RETURNING POINT ESTIMATE SINCE RIGHT IS NONE"
            return node.pointEstimate
    else:
        if node.left != None:
            # print "     Going Left: " + str(node.pointEstimate)
            return trainDataTest(node.left, docId, arrayToCheck)
        else:
            # print " RETURNING POINT ESTIMATE SINCE LEFT IS NONE"
            return node.pointEstimate

def decisionTreeLearner(completeDataSet, remainingFeatures, method):
    # Calculate the point estimate for all the articles
    rootNode = generateMaximumInformationGainElement(completeDataSet, remainingFeatures, method)
    rootNode.remainingFeaturesSet = calculateRemainingFeatures(remainingFeatures, rootNode.featureName)
    # print " DATA SPLIT : " + str(rootNode.informationGain)
    print " **************** TRAINING ACCURACY:    " + str(runTrainingDataAccuracy(rootNode, 0, method)) + " INDEX: " + str(0) + " **************** "
    print " **************** TESTING ACCURACY:    " + str(runTestDataAccuracyTest(rootNode, 0, method)) + " INDEX: " + str(0) + " **************** "

    rootNode.nodeIndex = 0

    # print " TESTING ACCURACY:    " + str(testAccuracy(rootNode)) + " INDEX: " + str(0)
    # print maxGain.remainingFeaturesSet
    heapq = PriorityQueue()
    heapq.push(rootNode, rootNode.informationGain)
    index = 0
    deltaGain = -1.0
    # TODO: Alter this condition to be more valid -- delta is zero
    while(index < 100):
        # print "     ********************** New Element **********************   " + str(index)
        topFeature = heapq.pop()

        # print wordDict[topFeature.featureName]
        # print topFeature.informationGain

        # for each feature 0 or 1
        for featureValue in range(0, 2):
            # print "--------------- Feature Iteration ---------------: remaining features = " + str(len(remainingFeatures))
            splitDataSet = topFeature.dataSets[featureValue]
            # print len(splitDataSet)
            # print topFeature.dataSets[1]
            # print len(topFeature.dataSets[1])
            newInformationGainNode = generateMaximumInformationGainElement(splitDataSet, topFeature.remainingFeaturesSet, method)
            newInformationGainNode.remainingFeaturesSet = calculateRemainingFeatures(topFeature.remainingFeaturesSet, newInformationGainNode.featureName)
            index += 1
            newInformationGainNode.nodeIndex = index

            # print " *************** NEW WORD ***************"
            # print newInformationGainNode.featureName
            # print "     NODE INDEX: " + str(newInformationGainNode.nodeIndex)
            # print wordDict[newInformationGainNode.featureName]
            # print newInformationGainNode.informationGain

            # if(index == 9):
            #     print "   INDEX 9 NEGATIVE: "
            #     print newInformationGainNode.dataSets[0]
            #     print str(len(newInformationGainNode.dataSets[0]))
            #     print "   INDEX 9 POSITIVE: "
            #     print newInformationGainNode.dataSets[1]
            #     print str(len(newInformationGainNode.dataSets[1]))
            #
            # print str(len(newInformationGainNode.dataSets[1]))
            #
            # # print newInformationGainNode.remainingFeaturesSet
            # print newInformationGainNode.pointEstimate
            # print " *************** ******* ***************"

            if(featureValue == 0):
                topFeature.left = newInformationGainNode
                labelEdge = "No"
            else:
                topFeature.right = newInformationGainNode
                labelEdge = "Yes"


            graph.node(str(index) + " " + wordDict[newInformationGainNode.featureName] + "\\n" + str(newInformationGainNode.informationGain) + "\\n" + str(newInformationGainNode.pointEstimate))

            graph.edge(str(topFeature.nodeIndex) + " " + wordDict[topFeature.featureName] + "\\n" + str(topFeature.informationGain) + "\\n" + str(topFeature.pointEstimate),
                       str(index) + " " + wordDict[newInformationGainNode.featureName] + "\\n" + str(newInformationGainNode.informationGain) + "\\n" + str(newInformationGainNode.pointEstimate), label=labelEdge)
            print " **************** TRAINING ACCURACY:    " + str(runTrainingDataAccuracy(rootNode, index, method)) + " INDEX: " + str(index) + " **************** "
            print " **************** TESTING ACCURACY:    " + str(runTestDataAccuracyTest(rootNode, index, method)) + " INDEX: " + str(index) + " **************** "

            heapq.push(newInformationGainNode, newInformationGainNode.informationGain)
    print " **************** TRAINING ACCURACY:    " + str(runTrainingDataAccuracy(rootNode, index, method)) + " INDEX: " + str(index) + " **************** "
    print " **************** TESTING ACCURACY:    " + str(runTestDataAccuracyTest(rootNode, index, method)) + " INDEX: " + str(index) + " **************** "

    return rootNode

def startProgram(remainingFeatures):
    # Pass in all the ID numbers for the articles
    weightedDecisionTree = decisionTreeLearner(range(1, numArticles + 1), remainingFeatures, "weighted")
    printDecisionTree("weighted")
    avgDecisionTree = decisionTreeLearner(range(1, numArticles + 1), remainingFeatures, "average")
    printDecisionTree("average")
    generateDataGraphs()

def printDecisionTree(name):
    graph.render(filename='dataset/'+name)

def generateDataGraphs():
    plt.figure(1)
    plt.subplot(211)
    plt.plot(trainWeightedPlotPointsIndex, trainWeightedPlotPointsPercent, 'r--', trainAveragePlotPointsIndex, trainAveragePlotPointsPercent, 'b--')
    plt.title("Training Data Accuracy: Weighted (red) and Average (blue) Information Gain")
    plt.subplot(212)
    plt.plot(testWeightedPlotPointsIndex, testWeightedPlotPointsPercent, 'b--', testAveragePlotPointsIndex, testAveragePlotPointsPercent, 'g--')
    plt.title("Test Data Accuracy: Weighted (blue) and Average (green) Information Gain")
    plt.show()

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
            # print str(x)  + " " + str(y)
            testArrayData[x-1][y-1] = 1
    trainingData.close()



def populateTrainingDataValues():
    # WordFile holds all the actual words
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

def main():
    populateTrainingDataValues()



if __name__ == "__main__": main()
# from treelib import Tree, Node
# from graphviz import Digraph
import copy
import numpy as np
import math
from DataNode import DataNode
from DataNodePriorityQueue import PriorityQueue

# This funciton calculated the delta I(e) for each feature based split dataSet
# Input: The dataSet split for if a feature appears/or not within an article
# Outout: The I(e) for given data set
def computeWeightedInformationGain(inputDataSet):
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


# computeTotalInfomationContent: can be used to calculate the total Information Content
# Input: Example dataset split on the occurence of a feature
# Output:
def computeInformationGainDelta(featureAppearsIn, featureDoesNotAppearIn):
    # print featureAppearsIn
    infoGainFeatureOne = computeWeightedInformationGain(featureAppearsIn)
    # print " infoGainFeatureOne: " + str(infoGainFeatureOne)
    # print featureDoesNotAppearIn
    infoGainFeatureTwo = computeWeightedInformationGain(featureDoesNotAppearIn)
    # print " infoGainFeatureOne: " + str(infoGainFeatureTwo)
    featureOneTotal = float(featureAppearsIn.__len__())
    featureTwoTotal = float(featureDoesNotAppearIn.__len__())
    totalElements = featureOneTotal + featureTwoTotal
    # print totalElements
    return (featureOneTotal/totalElements) * infoGainFeatureOne + (featureTwoTotal/totalElements) * infoGainFeatureTwo


# calculates teh initial point estimate, how many of each type of article there are
# Input: The dataSet containing current articleIndex values
# Output: a DataNode determining which types of article is most common
def pointEstimate(dataset):
    print "       Point Estimate :    "
    numAtheism, numGraphics = computeNumberOfTimesEachLabelAppears(dataset)
    print "    Atheism: " + str(numAtheism)
    print "    Graphics: " + str(numGraphics)
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
    numGraphics = numGraphics / total
    numAtheism = numAtheism / total
    print "graphics: " + str(numGraphics)
    print "atheism: " + str(numAtheism)
    if(numGraphics == 0 and numAtheism == 0):
        raise Exception("Both NumGraphics and numAtheism are 0")
    elif(numGraphics == 0):
        return - (0 + numAtheism * math.log(numAtheism, 2))
    elif(numAtheism == 0):
        return -(numGraphics * math.log(numGraphics, 2) + 0)
    else:
        return -(numGraphics * math.log(numGraphics, 2) + numAtheism * math.log(numAtheism, 2))


def generateMaximumInformationGainElement(exampleDataset, remainingFeatures):
    pq = PriorityQueue()
    print exampleDataset
    informationConstant = computeTotalInfomationContent(exampleDataset)
    print "INFORMATION GAIN: " + str(informationConstant) + "   REMAINING FEATURES: "

    print remainingFeatures
    for wordID in remainingFeatures:
        # print "AT INDEX: " + str(wordID ) + " WORD: " + wordDict[wordID]
        featureAffirmative, featureNegative = generateFeatureBasedSplitDataSets(exampleDataset, wordID)
        splitInformationGain = computeInformationGainDelta(featureAffirmative, featureNegative)
        deltaInformationGain = informationConstant - splitInformationGain
        # print deltaInformationGain
        # print featureAffirmative
        # print featureNegative
        pq.push(DataNode(wordID, deltaInformationGain, [featureNegative, featureAffirmative]), deltaInformationGain)
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

def decisionTreeLearner(completeDataSet, remainingFeatures):
    # Calculate the point estimate for all the articles
    initialPointEstimate = pointEstimate(completeDataSet)
    maxGain = generateMaximumInformationGainElement(completeDataSet, remainingFeatures)
    maxGain.remainingFeaturesSet = calculateRemainingFeatures(remainingFeatures, maxGain.featureName)
    # print maxGain.remainingFeaturesSet
    heapq = PriorityQueue()
    heapq.push(maxGain, maxGain.informationGain)
    index = 0
    deltaGain = -1.0
    #TODO: Alter this condition to be more valid -- delta is zero
    while(index < 10):
        print "     ********************** New Element **********************   " + str(index)
        topFeature = heapq.pop()
        print wordDict[topFeature.featureName]
        print topFeature.informationGain
        # for each feature 0 or 1
        for featureValue in range(0, 2):
            print "--------------- Feature Iteration ---------------: remaining features = " + str(len(remainingFeatures))
            splitDataSet = topFeature.dataSets[featureValue]
            print len(splitDataSet)
            # print topFeature.dataSets[1]
            # print len(topFeature.dataSets[1])
            newInformationGainNode = generateMaximumInformationGainElement(splitDataSet, topFeature.remainingFeaturesSet)
            newInformationGainNode.remainingFeaturesSet = calculateRemainingFeatures(topFeature.remainingFeaturesSet, maxGain.featureName)
            # print newInformationGainNode.featureName
            # print wordDict[newInformationGainNode.featureName]
            # print newInformationGainNode.informationGain
            # print newInformationGainNode.dataSets[0]
            # print newInformationGainNode.dataSets[1]
            # print newInformationGainNode.remainingFeaturesSet
            newPointEstimate = pointEstimate(splitDataSet)
            # newInformationGainNode.decisionTree.append(pointEstimate(splitDataSet))
              # print pointestimate
            heapq.push(newInformationGainNode, newInformationGainNode.informationGain)
        index += 1
        deltaGain = topFeature.informationGain
        # break

def populateDataValues():
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
    # Pass in all the ID numbers for the articles
    decisionTreeLearner(range(1,numArticles+1), remainingFeatures)

def main():
    populateDataValues()


if __name__ == "__main__": main()
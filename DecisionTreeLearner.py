# import numpy as np

# from treelib import Tree, Node
# from graphviz import Digraph
import math
from DataNode import DataNode
from DataNodePriorityQueue import PriorityQueue

# calculates teh initial point estimate, how many of each type of article there are
# Input:
# Output: a DataNode determining which types of article is most common
def pointEstimate():
    graphics = articlesArray.count("1")
    atheism = articlesArray.count("2")
    return DataNode(1) if (graphics > atheism) else DataNode(2)

# This function splits the dataSet based on if the word appears in the article or not
# Input: A word ID
# Output:
# 1) An array of all the articles in which the word appears
# 2) An Array of all the articles in which the word does not appear
def generateFeatureBasedDataSets(wordID):
    featureOccuresDataset = []
    featureDoesNotOccureDataset = []
    for articleIndex in range(0, numArticles):
        if(array[articleIndex][wordID] == 1):
            featureOccuresDataset.append(articleIndex+1)
        else:
            featureDoesNotOccureDataset.append(articleIndex+1)
    return (featureOccuresDataset, featureDoesNotOccureDataset)


#TODO: handle case when the information gain is 0 for both
# This funciton calculated the delta I(e) for each feature based split dataSet
# Input: The dataSet split for if a feature appears/or not within an article
# Outout: The I(e) for given data set
def computeIndividualInformationGain(inputDataSet):
    featureArticleAtheism = 0.0
    featureArticleGraphics = 0.0
    # print inputDataSet
    for articleNumber in inputDataSet:
        if (articlesArray[articleNumber-1] == "1"):
            featureArticleAtheism += 1
        else:
            featureArticleGraphics += 1
    totalElements = featureArticleAtheism + featureArticleGraphics
    if(totalElements == 0):
        return 0
    else:
        fractionAtheism = featureArticleAtheism / totalElements
        fractionGraphics = featureArticleGraphics / totalElements
        # print fractionAtheism
        # print fractionGraphics
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
    infoGainFeatureOne = computeIndividualInformationGain(featureAppearsIn)
    # print infoGainFeatureOne
    infoGainFeatureTwo = computeIndividualInformationGain(featureDoesNotAppearIn)
    # print infoGainFeatureTwo
    featureOneTotal = float(featureAppearsIn.__len__())
    featureTwoTotal = featureDoesNotAppearIn.__len__()
    totalElements = featureOneTotal + featureTwoTotal
    # print  totalElements
    return (featureOneTotal/totalElements) * infoGainFeatureOne + (featureTwoTotal/totalElements) * infoGainFeatureTwo


# computeTotalInfomationContent: can be used to calculate the total Information Content
def computeTotalInfomationContent(articleExamples):
    numGraphics = float(articleExamples.count("1"))
    numAtheism = float(articleExamples.count("2"))
    total = numAtheism + numGraphics
    numGraphics = numGraphics/total
    numAtheism = numAtheism/total
    return -(numGraphics * math.log(numGraphics, 2) + numAtheism * math.log(numAtheism ,2))

def generatePriorityQueue(inputDataSet):
    pq = PriorityQueue()
    informationConstant = computeTotalInfomationContent(inputDataSet)
    # print "Info Constant: " + str(informationConstant)
    for word in range(0, numWords):
        # print "AT INDEX: " + str(word + 1) + " WORD: " + wordDict[word + 1]
        featureAppearsIn, featureDoesNotAppearIn = generateFeatureBasedDataSets(word)
        # print featureAppearsIn, featureDoesNotAppearIn
        splitInformationGain = computeInformationGainDelta(featureAppearsIn, featureDoesNotAppearIn)
        # print "Info Split: " + str(splitInformationGain)
        # print "Priority: " + str(informationConstant - splitInformationGain)
        # break
        pq.push(DataNode(word + 1, informationConstant - splitInformationGain), informationConstant - splitInformationGain)
        # print pq
    return pq

# Core algorithm to generate the decision tree learner
# Input:
# Output: The final decision tree
def decisionTreeLearner():
    dt = pointEstimate()
    heapq = generatePriorityQueue(articlesArray)
    topFeature = heapq.pop()
    print topFeature.featureName
    print topFeature.informationGain
    index = 0
    while(index < 3566):
        nextFeature = heapq.pop()
        for j in range(0, 2):


        index += 1


def populateDataValues():
    global array
    array = [[0 for x in range(numWords)] for y in range(numArticles)]
    # array = np.zeros((words, articles))

    testData = []
    with open('dataset/trainData.txt') as trainingData:
        testData = trainingData.readlines()

        for pair in testData:
            # print pair
            element = pair.split("\t")
            indexPairs = [int(index) for index in element]
            x, y = indexPairs
            # print str(x) + " and " + str(y)
            array[x-1][y-1] = 1

    trainingData.close()
    decisionTreeLearner()


def main():
    # WordFile holds all the actual words
    with open('dataset/words.txt') as wordFile:
        wordsArray = wordFile.read().splitlines()
        global wordDict
        wordDict = {}
        wordDict[0] = None
        index = 1
        for word in wordsArray:
            word.rstrip("\n") + " "
            wordDict[index] = word
            index += 1

    wordFile.close()
    global numWords
    numWords = len(wordsArray)

    # Classification of all the documents
    with open('dataset/trainLabel.txt') as articleFile:
        global articlesArray
        articlesArray = articleFile.read().splitlines()
        for article in articlesArray:
            article.rstrip("\n")
    articleFile.close()

    global numArticles
    numArticles = len(articlesArray)

    populateDataValues()


if __name__ == "__main__": main()
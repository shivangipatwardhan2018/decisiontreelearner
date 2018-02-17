
class DataNode:

    def __init__(self, featurename =None, informationgain=None, datasets=None, remainingfeatures = None, decisiontree=None):
        self.featureName = featurename
        self.informationGain = informationgain
        self.dataSets = datasets
        self.remainingFeaturesSet = remainingfeatures
        self.decisionTree = decisiontree




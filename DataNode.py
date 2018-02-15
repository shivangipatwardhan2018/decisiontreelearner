
class DataNode:

    # def __init__(self, featurename, informationgain=None, affirmativedataset = None, negativedataset = None, decisiontree = None):
    def __init__(self, featurename, informationgain=None, datasets=None, negativedataset=None,
                 decisiontree=None):
        self.featureName = featurename
        self.informationGain = informationgain
        self.dataSets = datasets
        # self.affirmativeDataSet = affirmativedataset
        # self.negativeDataSet = negativedataset
        self.decisionTree = decisiontree




class DataNode:

    def __init__(self, featurename =None, informationgain=None, datasets=None, pointestimate=None, remainingfeatures = None, leftnode=None, rightnode = None, nodeindex=None):
        self.featureName = featurename
        self.informationGain = informationgain
        self.dataSets = datasets
        self.remainingFeaturesSet = remainingfeatures
        self.pointEstimate = pointestimate
        self.left = leftnode
        self.right = rightnode
        self.nodeIndex = nodeindex




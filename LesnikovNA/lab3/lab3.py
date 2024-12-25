import cv2
import numpy as np
import os
import sys
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import argparse
from sklearn.utils import shuffle

Width = 200
Height = 200

class Classifier:
    def __init__(self, Classifier='svc'):
        if Classifier == "randomForest":
            self.Model = RandomForestClassifier(n_estimators=100)
        else:
            self.Model = SVC(probability=True, kernel='linear', random_state=23)
        self.Scaler = StandardScaler()

    def Train(self, X_train, Y_train):
        XTrainScaled = self.Scaler.fit_transform(X_train)
        self.Model.fit(XTrainScaled, Y_train)

    def Stats(self, X_test, Y_test):
        XTestScaled = self.Scaler.transform(X_test)
        YPred = self.Model.predict(XTestScaled)
        YProb = self.Model.predict_proba(XTestScaled)[:, 1]
        Accuracy = accuracy_score(Y_test, YPred)
        return Accuracy, YPred, YProb

def PlotClassificationResults(YTrue, YPred, ClassNames):
    Cm = confusion_matrix(YTrue, YPred)
    CatsCount = Cm[0, 0].sum()
    DogsCount = Cm[1, 1].sum()
    MisclassifiedCount = Cm[0, 1] + Cm[1, 0]
    Counts = [CatsCount, DogsCount, MisclassifiedCount]
    Labels = ['Cats', 'Dogs', 'Wrong']
    Colors = ['green', 'green', 'red']
    plt.figure(figsize=(8, 6))
    plt.bar(Labels, Counts, color=Colors)
    plt.ylabel('Number of images')
    for i, count in enumerate(Counts):
        plt.text(i, count + 1, str(count), ha='center', fontsize=12)
    plt.tight_layout()
    plt.show()

def VisualizeFeatures(Images, Descriptor):
    for i, Img in enumerate(Images[:5]):
        Gray = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)
        KeyPoints = Descriptor.detect(Gray, None)
        ImgWithKeyPoints = cv2.drawKeypoints(Gray, KeyPoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        plt.figure(figsize=(6, 6))
        plt.imshow(ImgWithKeyPoints, cmap='gray')
        plt.title(f"Image {i + 1}: {len(KeyPoints)} keypoints detected")
        plt.axis('off')
        plt.show()

def PlotDescriptorsHistogram(CatsDescriptors, DogsDescriptors, KMeans, NClusters):
    AllCatsDescriptors = np.vstack([Desc for Desc in CatsDescriptors if Desc is not None])
    AllDogsDescriptors = np.vstack([Desc for Desc in DogsDescriptors if Desc is not None])

    CatsPredictions = KMeans.predict(AllCatsDescriptors)
    DogsPredictions = KMeans.predict(AllDogsDescriptors)

    CatsHistogram = np.bincount(CatsPredictions, minlength=NClusters)
    DogsHistogram = np.bincount(DogsPredictions, minlength=NClusters)

    plt.figure(figsize=(14, 7))

    plt.subplot(1, 2, 1)
    plt.bar(range(NClusters), CatsHistogram, color='skyblue')
    plt.xlabel('Cluster Index')
    plt.ylabel('Number of Descriptors')
    plt.title(f'Histogram of Descriptors Distribution for Cats')

    plt.subplot(1, 2, 2)
    plt.bar(range(NClusters), DogsHistogram, color='orange')
    plt.xlabel('Cluster Index')
    plt.ylabel('Number of Descriptors')
    plt.title(f'Histogram of Descriptors Distribution for Dogs')

    plt.tight_layout()
    plt.show()

def PlotConfusionMatrix(YTrue, YPred, ClassNames):
    Cm = confusion_matrix(YTrue, YPred)
    plt.figure(figsize=(8, 6))
    plt.imshow(Cm, interpolation='nearest', cmap='coolwarm')
    plt.title('Confusion Matrix')
    plt.colorbar()
    TickMarks = np.arange(len(ClassNames))
    plt.xticks(TickMarks, ClassNames)
    plt.yticks(TickMarks, ClassNames)
    Thresh = Cm.max() / 2.
    for i, j in np.ndindex(Cm.shape):
        plt.text(j, i, format(Cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if Cm[i, j] > Thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def ParseArguments():
    Parser = argparse.ArgumentParser()
    Parser.add_argument('-extractAlg', type=str, choices=['sift', 'orb'], dest='Descriptor')
    Parser.add_argument('-classifier', type=str, choices=['svc', 'randomForest'], dest='Classifier')
    Parser.add_argument('-train', type=str, dest='TrainDir')
    Parser.add_argument('-test', type=str, dest='TestDir')
    Parser.add_argument('-clusters', type=int, dest='NClusters')
    Parser.add_argument('-imagesCount', type=int, dest='ImagesCount')
    Args = Parser.parse_args()
    return Args

def LoadImages(Folder, Label, MaxCount):
    Images = []
    Labels = []
    for Filename in os.listdir(Folder):
        if MaxCount < 0:
            break
        MaxCount -= 1
        Path = os.path.join(Folder, Filename)
        Img = cv2.imread(Path)
        if Img is not None:
            Img = cv2.resize(Img, (Width, Height))
            Images.append(Img)
            Labels.append(Label)
    return Images, Labels

class Extractor:
    def __init__(self, NClusters, Descriptor):
        self.NClusters = NClusters
        self.KMeans = KMeans(n_clusters=self.NClusters, random_state=23)
        if Descriptor == 'sift':
            self.Detector = cv2.SIFT_create()
        elif Descriptor == 'orb':
            self.Detector = cv2.ORB_create()
        else:
            raise ValueError(f"Unknown extraction algorithm")

    def ExtractFeatures(self, Images):
        DescriptorsList = []
        KeyPointsCounts = []
        for Img in Images:
            Gray = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)
            KeyPoints, Descriptors = self.Detector.detectAndCompute(Gray, None)
            if Descriptors is not None:
                DescriptorsList.append(Descriptors)
                KeyPointsCounts.append(len(KeyPoints))
        self.AverageKeyPoints = np.mean(KeyPointsCounts) if KeyPointsCounts else 0
        return DescriptorsList

    def ComputeBowHistograms(self, Images):
        Features = []
        for Img in Images:
            Gray = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)
            KeyPoints, Descriptors = self.Detector.detectAndCompute(Gray, None)
            Histogram = np.zeros(self.NClusters)
            if Descriptors is not None:
                Predictions = self.KMeans.predict(Descriptors)
                for Pred in Predictions:
                    Histogram[Pred] += 1
            Features.append(Histogram)
        return np.array(Features)

    def FitKMeansClusterization(self, DescriptorsList):
        AllDescriptors = []
        for Desc in DescriptorsList:
            if Desc is not None:
                AllDescriptors.append(Desc)
        AllDescriptors = np.vstack(AllDescriptors)
        self.KMeans.fit(AllDescriptors)

def main():
    Args = ParseArguments()
    Data = Extractor(NClusters=Args.NClusters, Descriptor=Args.Descriptor)

    CatsTrain, YCatsTrain = LoadImages(os.path.join(Args.TrainDir, 'Cat'), 0, Args.ImagesCount)
    DogsTrain, YDogsTrain = LoadImages(os.path.join(Args.TrainDir, 'Dog'), 1, Args.ImagesCount)
    CatsTest, YCatsTest = LoadImages(os.path.join(Args.TestDir, 'Cat'), 0, Args.ImagesCount / 3)
    DogsTest, YDogsTest = LoadImages(os.path.join(Args.TestDir, 'Dog'), 1, Args.ImagesCount / 3)

    TrainImages = CatsTrain + DogsTrain
    TrainLabels = YCatsTrain + YDogsTrain
    TestImages = CatsTest + DogsTest
    TestLabels = YCatsTest + YDogsTest
    TrainImages, TrainLabels = shuffle(TrainImages, TrainLabels, random_state=42)
    TestImages, TestLabels = shuffle(TestImages, TestLabels, random_state=42)
    TrainDescriptors = Data.ExtractFeatures(TrainImages)
    Data.FitKMeansClusterization(TrainDescriptors)

    print(f"Average number of control points: ", Data.AverageKeyPoints)
    VisualizeFeatures(TrainImages, Data.Detector)
    CatsDescriptors = Data.ExtractFeatures([CatsTrain[10]])
    DogsDescriptors = Data.ExtractFeatures([DogsTrain[10]])
    PlotDescriptorsHistogram(CatsDescriptors, DogsDescriptors, Data.KMeans, Data.NClusters)
    XTrainFeatures = Data.ComputeBowHistograms(TrainImages)
    XTestFeatures = Data.ComputeBowHistograms(TestImages)

    Model = Classifier(Args.Classifier)
    Model.Train(XTrainFeatures, TrainLabels)
    TrainAccuracy, YTrainPred, YTrainProb = Model.Stats(XTrainFeatures, TrainLabels)
    TestAccuracy, YTestPred, YTestProb = Model.Stats(XTestFeatures, TestLabels)
    print("Train Accuracy:", TrainAccuracy)
    print("Test Accuracy:", TestAccuracy)
    ClassNames = ['Cats', 'Dogs']
    PlotConfusionMatrix(TestLabels, YTestPred, ClassNames)
    PlotClassificationResults(TestLabels, YTestPred, ClassNames)
    
if __name__ == "__main__":
    sys.exit(main() or 0)

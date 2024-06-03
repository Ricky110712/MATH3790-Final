import numpy as np
import cv2
import matplotlib.pyplot as plt

def loadImageSet():
    FaceMat = np.zeros((15, 100 * 100))  # Initialize a matrix to store 15 images of 100x100 pixels.
    file = "./Dataset/"
    j = 0
    try:
        for i in range(6, 161, 11):
            tmp = str(i)
            img = cv2.imread(file + "s" + tmp + ".bmp", 0)  # Read as a grayscale image.
            FaceMat[j, :] = img.flatten()  # Flatten the image and store it in the matrix.
            j += 1
    except Exception as e:
        print(f"Error loading image: {e}")

    print(f"Number of images loaded: {j}")
    print(f"Shape of FaceMat: {FaceMat.shape}")
    return FaceMat

def computeEigenVectors(FaceMat, selecthr=0.8):
    FaceMat = FaceMat.T  # Transpose the matrix to get 10000x15.
    aveImg = np.mean(FaceMat, axis=1)  # Calculate the average face image.
    diffTrain = FaceMat - aveImg[:, None]  # Subtract the average face image from each face image.

    tempDiffTrain = diffTrain.T @ diffTrain  # Calculate the covariance matrix.

    eigvals, eigVects = np.linalg.eig(tempDiffTrain)  # Calculate eigenvalues and eigenvectors.

    eigSortIndex = np.argsort(-eigvals)  # Sort eigenvalues in descending order.

    tempArrage = FaceMat.shape[1]
    for i in range(tempArrage):
        tempArryI = eigSortIndex[:i]
        if (eigvals[tempArryI] / eigvals.sum()).sum() >= selecthr:
            eigSortIndex = tempArryI
            break

    tempEigSortIndex = eigVects[:, eigSortIndex]
    covVects = diffTrain @ tempEigSortIndex  # Select corresponding eigenvectors.

    print(f"Shape of CovVects: {covVects.shape}")
    return aveImg, covVects, diffTrain

def judgeFace(judgeImg, FaceVector, avgImg, diffTrain):
    diff = judgeImg.T - avgImg  # Calculate the difference from the average face.
    weiVec = FaceVector.T @ diff  # Project the difference into PCA space.

    res = 0
    resVal = np.inf
    for i in range(15):
        TrainVec = FaceVector.T @ diffTrain[:, i]
        tempVal = np.sum((weiVec - TrainVec) ** 2)
        if tempVal < resVal:
            res = i
            resVal = tempVal

    tempRes = res + 1
    return tempRes

def calculateAccuracy(FaceVector, avgImg, diffTrain, nameList, characteristic):
    fileArry = "./Dataset/"
    accuracies = []
    countSum = 0

    for P, K in enumerate(range(1, 12)):
        cout = 0
        for N, M in enumerate(range(K, 166, 11)):
            try:
                loadname = fileArry + "s" + str(M) + ".bmp"
                judgeImg = cv2.imread(loadname, 0)
                tempMat = judgeImg.flatten()
                if judgeFace(tempMat, FaceVector, avgImg, diffTrain) == int(nameList[N]):
                    cout += 1
                    countSum += 1
            except Exception as e:
                print(f"Error processing image {M}: {e}")

        accuracy = float(cout) / len(nameList)
        accuracies.append(accuracy)
        character = characteristic[P] if P < len(characteristic) else 'unknown'
        print(f'Accuracy of {character} is {accuracy:.6f}')

    overall_accuracy = float(countSum) / 165
    print(f'Overall accuracy is {overall_accuracy:.6f}')
    return accuracies, overall_accuracy

def plotAccuracies(characteristic, accuracies):
    plt.figure(figsize=(10, 6))
    plt.bar(characteristic, accuracies, color='skyblue')
    plt.xlabel('Characteristics')
    plt.ylabel('Accuracy')
    plt.title('Face Recognition Accuracy by Characteristics')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    avgImg, FaceVector, diffTrain = computeEigenVectors(loadImageSet(), selecthr=0.9)
    nameList = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15']
    characteristic = ['centerlight', 'glasses', 'happy', 'leftlight', 'noglasses', 'unknown', 'rightlight', 'sad', 'sleepy',
                      'surprised', 'wink']

    accuracies, overall_accuracy = calculateAccuracy(FaceVector, avgImg, diffTrain, nameList, characteristic)
    plotAccuracies(characteristic, accuracies)

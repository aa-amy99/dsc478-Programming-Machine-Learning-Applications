'''
Modiflied for HW 3 by Amy Aumpansub (add function Cos_Sim and findTerms)

Cannnot import this module to Jupyter Notebook bcos of the difference in Python Version

So, I copied those following functions and pasted in Jupyter Notebook 
'''
import numpy as np
from numpy import *

def distEuclid(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2))) #la.norm(vecA-vecB)

def Cos_Sim(x,D):# where D and x is are array inputs from Kmeans function in part C
    
    normX = np.linalg.norm(x)
    normD = np.linalg.norm(D)
    CosSim = np.dot(D,x)/(normD * normX )
    dist = 1 - CosSim
    return dist

def randCent(dataSet, k):
	n = shape(dataSet)[1]
	centroids = zeros((k,n), dtype=float)
	for j in range(n): #create random cluster centers
		minJ = min(dataSet[:,j])
		rangeJ = float(max(dataSet[:,j]) - minJ)
		centroids[:,j] = minJ + rangeJ * random.rand(k)
	return centroids 

def kMeans(dataSet, k, distMeas=distEuclid, createCent=randCent):
    m = shape(dataSet)[0]
    clusterAssment = zeros((m,2))#create mat to assign data points 
                                      #to a centroid, also holds SE of each point
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):#for each data point assign it to the closest centroid
            minDist = inf; minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI; minIndex = j
            if clusterAssment[i,0] != minIndex: clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2
        # print centroids
        for cent in range(k):#recalculate centroids
            ptsInClust = dataSet[nonzero(clusterAssment[:,0]==cent)[0]] #get all the point in this cluster - Note: this was incorrect in the original distribution.
            if(len(ptsInClust)!=0):
		centroids[cent,:] = mean(ptsInClust, axis=0) #assign centroid to mean - Note condition was added 10/28/2013
    return centroids, clusterAssment

def biKmeans(dataSet, k, distMeas=distEuclid):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))
    centroid0 = mean(dataSet, axis=0).tolist()[0]
    centList =[centroid0] #create a list with one centroid
    for j in range(m): #calc initial Error
        clusterAssment[j,1] = distMeas(mat(centroid0), dataSet[j,:])**2
    while (len(centList) < k):
        lowestSSE = inf
        for i in range(len(centList)):
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A==i)[0],:] #get the data points currently in cluster i
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            sseSplit = sum(splitClustAss[:,1])#compare the SSE to the currrent minimum
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1])
            print "sseSplit, and notSplit: ",sseSplit,sseNotSplit
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        bestClustAss[nonzero(bestClustAss[:,0] == 1)[0],0] = len(centList) #change 1 to 3,4, or whatever
        bestClustAss[nonzero(bestClustAss[:,0] == 0)[0],0] = bestCentToSplit
        print 'the bestCentToSplit is: ',bestCentToSplit
        print 'the len of bestClustAss is: ', len(bestClustAss)
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]#replace a centroid with two best centroids 
        centList.append(bestNewCents[1,:].tolist()[0])
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:]= bestClustAss#reassign new clusters, and SSE
    return mat(centList), clusterAssment

def findTerms(Data, k, N):
    myCentroids, clustAssing = kMeans(Data,k)
    for i in range(0,k,1):
        print("CLUSTER: %d   results" %(i))
        clusterNum = clustAssing[:,0]#Clustering Number
        clusterDT = Data[clusterNum == i]#Create Matrix for each cluster (DT)
        numDocs =  clusterDT.shape[0]#add row of each cluster Matrix
        print ("Number of documents in cluster : %8d   " %(numDocs))
        DocsFreq = np.array([(clusterDT.T!=0).sum(1)]).T
        freqPercent = DocsFreq/float(numDocs)
        Results = np.concatenate((Label_DF, DocsFreq, freqPercent), axis=1)#each element = [percentFreq, freq, term]
        sortedResults = np.flip(Results[Results[:,-1].argsort()])#Sorted by decesending order
        print ("        Word      DF    PercentofDocs")
        for i, result in enumerate(sortedResults[:N]):
            term = result[2]
            DF = result[1]
            freqPer = (result[0]*100)
            print ("%12s   %5d   %14.6f  "  %(term, DF , freqPer) 
    return myCentroids, clustAssing



from numpy import *
from numpy import linalg as la
import numpy as np

'''
Modified by Amy Aumpansub
Test function and print_most_similar_jokes function is added

'''

def ecludSim(inA,inB):
    return 1.0 / (1.0 + la.norm(inA - inB))

def pearsSim(inA,inB):
    if len(inA) < 3 : return 1.0
    return 0.5 + 0.5 * corrcoef(inA, inB, rowvar = 0)[0][1]

def cosSim(inA,inB):
    num = float(inA.T * inB)
    denom = la.norm(inA)*la.norm(inB)
    return 0.5 + 0.5 * (num / denom)

def standEst(dataMat, user, simMeas, item):
    n = shape(dataMat)[1]
    simTotal = 0.0; ratSimTotal = 0.0
    for j in range(n):
        userRating = dataMat[user,j]
        if userRating == 0: continue
        overLap = nonzero(logical_and(dataMat[:,item]>0, \
                                      dataMat[:,j]>0))[0]
        if len(overLap) == 0: similarity = 0
        else: similarity = simMeas(dataMat[overLap,item], \
                                   dataMat[overLap,j])
        #print 'the %d and %d similarity is: %f' % (item, j, similarity)
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0: return 0
    else: return ratSimTotal/simTotal
    
def svdEst(dataMat, user, simMeas, item):
    n = shape(dataMat)[1]
    simTotal = 0.0; ratSimTotal = 0.0
    data=mat(dataMat)
    U,Sigma,VT = la.svd(data)
    Sig4 = mat(eye(4)*Sigma[:4]) #arrange Sig4 into a diagonal matrix
    xformedItems = data.T * U[:,:4] * Sig4.I  #create transformed items
    for j in range(n):
        userRating = data[user,j]
        if userRating == 0 or j==item: continue
        similarity = simMeas(xformedItems[item,:].T,\
                             xformedItems[j,:].T)
        #print 'the %d and %d similarity is: %f' % (item, j, similarity)
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0: return 0
    else: return ratSimTotal/simTotal

# This function is not needed for Assignment 4, but may be useful for experimentation
def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=standEst):
    unratedItems = nonzero(dataMat[user,:].A==0)[1] #find unrated items 
    if len(unratedItems) == 0: return 'you rated everything'
    itemScores = []
    for item in unratedItems:
        estimatedScore = estMethod(dataMat, user, simMeas, item)
        itemScores.append((item, estimatedScore))
    return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[:N]

# This function performs evaluatoin on a single user based on the test_ratio
# For example, with test_ratio = 0.2, a randomly selected 20 percent of rated 
# items by the user are withheld and the rest are used to estimate the withheld ratings

def cross_validate_user(dataMat, user, test_ratio, estMethod=standEst, simMeas=pearsSim):
    number_of_items = np.shape(dataMat)[1]
    rated_items_by_user = np.array([i for i in range(number_of_items) if dataMat[user,i]>0])
    test_size = test_ratio * len(rated_items_by_user)
    test_size = int(test_size)
    test_indices = np.random.randint(0, len(rated_items_by_user), test_size)
    withheld_items = rated_items_by_user[test_indices]
    original_user_profile = np.copy(dataMat[user])
    dataMat[user, withheld_items] = 0 # So that the withheld test items is not used in the rating estimation below
    error_u = 0.0
    count_u = len(withheld_items)

    # Compute absolute error for user u over all test items
    for item in withheld_items:
        # Estimate rating on the withheld item
        estimatedScore = estMethod(dataMat, user, simMeas, item)
        error_u = error_u + abs(estimatedScore - original_user_profile[item])	

    # Now restore ratings of the withheld items to the user profile
    for item in withheld_items:
        dataMat[user, item] = original_user_profile[item]

    # Return sum of absolute errors and the count of test cases for this user
    # Note that these will have to be accumulated for each user to compute MAE
    return error_u, count_u

def test(dataMat, test_ratio, estMethod):
        # Write this function to iterate over all users and for each perform evaluation by calling
        # the above cross_validate_user function on each user. MAE will be the ratio of total error 
        # across all test cases to the total number of test cases, for all users
        totalError = 0
        totalNum = 0
        MAE = 0
        for user in range(dataMat.shape[0]):#iterate over the row
            if estMethod == "svdEst":
                error, count = cross_validate_user(dataMat, user, test_ratio, estMethod=svdEst)
            elif estMethod == "standEst":
                error, count = cross_validate_user(dataMat, user, test_ratio, estMethod=standEst)
            totalError = totalError + error
            totalNum = totalNum + count
        MAE = totalError/totalNum 
        print ('Mean Absoloute Error for ',estMethod,' : ', MAE)

def print_most_similar_jokes(dataMat, jokes, queryJoke, k, metric=pearsSim):
    
    # Write this function to find the k most similar jokes (based on user ratings) to a queryJoke
    # The queryJoke is a joke id as given in the 'jokes.csv' file (an corresponding to the a column in dataMat)
    # You must compare ratings for the queryJoke (the column in dataMat corresponding to the joke), to all
    # other joke rating vectors and return the top k. Note that this is the same as performing KNN on the 
        # columns of dataMat. The function must retrieve the text of the joke from 'jokes.csv' file and print both
    # the queryJoke text as well as the text of the returned jokes.
    
        #Transpose DataMat to (row = user, column = jokes)
        dataMatT = dataMat.T
        if metric == "cosSim":
            dataMatT = dataMatT.T
        numUser = dataMatT.shape[0]
        OneQueryJoke = dataMatT[queryJoke]
    
        totalSim = np.zeros((numUser, 1))#use array to store similarity value for (100 users)
        indexSim = np.zeros((numUser, 1))
        recommendjoke=[]

        #Iterate over each row (user)
        i = 0
        while i < numUser:
            Sim = metric(dataMatT[i],OneQueryJoke)#calculate similarity value
            if Sim ==1:#account for Selected Joke
                totalSim[i]=0
                indexSim[i] = i
            else:
                totalSim[i] = Sim
                indexSim[i] = i
            i += 1
        Results = np.concatenate((totalSim,indexSim), axis =1)
        sortedResults = np.flip(Results[Results[:,0].argsort()])
        indexJoke = sortedResults[:, 0].astype(int)
        for index in indexJoke[:k]:
            recommendjoke.append(index)
    
        print("Selected Joke: \n %s" %jokes[queryJoke])
    
        print ("Top %d Recommended Joke: \n" %k)
        for joke in recommendjoke:
            print(jokes[joke])
            print("________________________________")

def load_jokes(file):
    jokes = genfromtxt(file, delimiter=',', dtype=str)
    jokes = np.array(jokes[:,1])
    return jokes

def get_joke_text(jokes, id):
    return jokes[id]

# dataMat = genfromtxt('modified_jester_data.csv',delimiter=',')

# test(dataMat, 0.2, svdEst)
# test(dataMat, 0.2, standEst)

# jokes = load_jokes('jokes.csv')
# print_most_similar_jokes(dataMat, jokes, 3, 5, pearsSim)

''' See example output below:

Selected joke: 

Q. What's the difference between a man and a toilet? A. A toilet doesn't follow you around after you use it.

Top 5 Recommended jokes are :

Q: What's the difference between a Lawyer and a Plumber? A: A Plumber works to unclog the system. 
_______________
What do you call an American in the finals of the world cup? "Hey Beer Man!" 
_______________
Q. What's 200 feet long and has 4 teeth? <P>A. The front row at a Willie Nelson Concert. 
_______________
A country guy goes into a city bar that has a dress code and the maitred' demands he wear a tie. Discouraged the guy goes to his car to sulk when inspiration strikes: He's got jumper cables in the trunk! So he wrapsthem around his neck sort of like a string tie (a bulky string tie to be sure) and returns to the bar. The maitre d' is reluctant but says to the guy "Okay you're a pretty resourceful fellow you can come in... but just don't start anything"!   
_______________
What do you get when you run over a parakeet with a lawnmower? <P>Shredded tweet. 
_______________

'''


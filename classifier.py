#Caelan Collins Pitch Type Classifier

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

def main():
    #load CSV data into pandas data frames
    dfTrain = pd.read_csv('train.csv')
    dfTest = pd.read_csv('test.csv')

    #drop features from data frames
    features_to_drop = ['last','first','mlbid','ab_id','pitch_id','inning','stand','height','x0','z0','px','pz','count']
    finalTestingData = dfTest.drop(features_to_drop, axis=1)

    testingData = dfTest.drop(features_to_drop,axis=1)

    #also remove pitchtype from training data
    features_to_drop += ['pitch_type']
    X = dfTrain.drop(features_to_drop,axis=1)
    y = dfTrain['pitch_type']

    #select and fit classification algorithm
    clf = RandomForestClassifier()
    clf.fit(X,y)

    #predict pitch types of given test data
    pitchPredictions = clf.predict(testingData)
    dfTest['pitch_type_predictions'] = pitchPredictions
    dfTest.to_csv('TestWithPredictions.csv', index=False)

if __name__== "__main__":
    main()




#createCharts(dfTrain,pitchTypes)

import tensorflow as tf
import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from numpy.random import seed
np.random.seed(2)
tf.random.set_seed(2)

# Hyper Parameter
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
BATCH_SIZE = 64
NUM_FOLDS = 10
RUNS= 3
TOTALEPOCHS = NUM_FOLDS*NUM_EPOCHS*RUNS




np.set_printoptions(suppress=True)

def createList(r1, r2): 
  
    if (r1 == r2): 
        return r1 
  
    else: 
  
        res = [] 
  
        while(r1 < r2+1 ): 
              
            res.append(r1) 
            r1 += 1
        return res 
        

def SameEpochT(arr,num_epoch,total_epochs):
    new = []
    for i in range(total_epochs):
        new.append(arr[i::num_epoch])   
    return new


def MeanAcrossEpochT(arr,num_epoch,num_folds,runs,total_epochs):
    
    tot = []
    arr = SameEpochT(arr,num_epoch,total_epochs)  

    for i in range(num_epoch):
        val = 0
        val = np.sum(arr[i])/(RUNS*num_folds)
        tot.append(val)
    return tot     


def StdAcrossEpochT(arr,num_epoch,num_folds,runs,total_epochs):
    
    tot = []
    stdarr = []  
    new = []
    for i in range(num_epoch):      
        val = 0
        val = np.std(arr[i::num_epoch],ddof=1)
        tot.append(val)
    return tot             

def make_dataset(features,labels,n_split):

    def gen():

        kfold = KFold(n_splits=n_split, shuffle=True, random_state=74)
        for train, test in kfold.split(features, labels):
            X_train, X_test = features[train], features[test]
            y_train, y_test= labels[train], labels[test] 
            yield X_train, y_train, X_test, y_test
   
    return tf.data.Dataset.from_generator(gen, (tf.float64,tf.float64,tf.float64,tf.float64))  



def replace_nan(x):
    if x == " ":
        return np.nan
    else:
        return float(x)

def remove_nan_values(x):
    return [e for e in x if not math.isnan(e)]       

Data= pd.read_csv("C:/Users/AutoMAttic/Desktop/honours/COS711/Assignments/Assignment3/A3 Data/A3 Data/Train.csv")



features = ["temp","precip","rel_humidity","wind_dir","wind_spd","atmos_press"]   



for feature in features : 
    Data[feature]=Data[feature].apply(lambda x: [ replace_nan(X) for X in x.replace("nan"," ").split(",")])
    


Data.columns.tolist()


for feature in features:
    Data[feature]=Data[feature].apply(remove_nan_values)    


lab  = LabelEncoder()
lab.fit(Data["location"])
Data['location'] = lab.transform(Data["location"])



for x in range(121):
    Data["newtemp"+ str(x)] = Data.temp.str[x]
    Data["newprecip"+ str(x)] = Data.precip.str[x]
    Data["newrel_humidity"+ str(x)] = Data.rel_humidity.str[x]
    Data["newwind_dir"+ str(x)] = Data.wind_dir.str[x]
    Data["windspeed"+ str(x)] = Data.wind_spd.str[x]
    Data["atmospherepressure"+ str(x)] = Data.atmos_press.str[x]


Data.drop(features,1,inplace=True)



X=Data.drop(["ID",'location','target'],axis =1)

X= np.nan_to_num(X,copy=True, nan=-999)
y = Data.target

# Scale features
scaler = StandardScaler().fit(X)
X = scaler.transform(X)

X= np.reshape(X,(-1,726,1))
opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)


TrainResults = []
Testresults =[]
test = []
fold =1

samp =[]
lastepochmean =[]

for run in range(RUNS):
     
    seed = run + 3
    tf.random.set_seed(seed)
    np.random.seed(seed)

    #model
    model = tf.keras.Sequential([
                              
                              tf.keras.layers.Conv1D(filters=20, kernel_size=6, activation='relu', input_shape=(726,1)),
                              tf.keras.layers.Conv1D(filters=20, kernel_size=6, activation='relu'),
                              tf.keras.layers.MaxPool1D(pool_size=6),
                              tf.keras.layers.Conv1D(filters=10, kernel_size=2, activation='relu'),
                              tf.keras.layers.Conv1D(filters=10, kernel_size=2, activation='relu'),
                              tf.keras.layers.MaxPool1D(pool_size=2),
                              tf.keras.layers.Flatten(),
                              tf.keras.layers.Dense(128, activation='relu'),
                              tf.keras.layers.Dense(64, activation='sigmoid'),
                              tf.keras.layers.Dense(1)
    ])
    

   #Save weights 
    Wsave = model.get_weights()
    model.compile(optimizer=opt, loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])
    print("run {:d}".format(run+1))
    datasets = make_dataset(X,y,NUM_FOLDS)
    run = run+1
    fold =1
   
    for X_train,y_train,X_test,y_test in datasets:
        
        
        model.set_weights(Wsave)
        print("Fold {:d}".format(fold))
        fold = fold+1
        Results=model.fit(X_train, y_train, epochs=NUM_EPOCHS, shuffle = False, validation_data=(X_test, y_test))
        TrainResults.append(Results.history)
        print('\n')

   
   




TrainLossResults = [d['loss'] for d in TrainResults]
TrainRMSEresults = [d['root_mean_squared_error'] for d in TrainResults]

TestLossResults = [d['val_loss'] for d in TrainResults]
TestRMSEresults = [d['val_root_mean_squared_error'] for d in TrainResults]


TrainLossResults = np.concatenate( TrainLossResults , axis=0 )
TrainRMSEresults = np.concatenate( TrainRMSEresults , axis=0 )

TestLossResults = np.concatenate( TestLossResults , axis=0 )
TestRMSEresults  = np.concatenate( TestRMSEresults  , axis=0 )



atrstdLoss = []
atestdLoss = []
atrstdLoss = StdAcrossEpochT(TrainLossResults,NUM_EPOCHS,NUM_FOLDS,RUNS,TOTALEPOCHS)
atestdLoss = StdAcrossEpochT(TestLossResults,NUM_EPOCHS,NUM_FOLDS,RUNS,TOTALEPOCHS)


atrstdRMSE = []
atestdRMSE = []
atrstdRMSE = StdAcrossEpochT(TrainRMSEresults,NUM_EPOCHS,NUM_FOLDS,RUNS,TOTALEPOCHS)
atestdRMSE = StdAcrossEpochT(TestRMSEresults,NUM_EPOCHS,NUM_FOLDS,RUNS,TOTALEPOCHS)




TrainLossResults= MeanAcrossEpochT(TrainLossResults,NUM_EPOCHS,NUM_FOLDS,RUNS,TOTALEPOCHS)
TrainRMSEresults= MeanAcrossEpochT(TrainRMSEresults,NUM_EPOCHS,NUM_FOLDS,RUNS,TOTALEPOCHS)

TestLossResults= MeanAcrossEpochT(TestLossResults,NUM_EPOCHS,NUM_FOLDS,RUNS,TOTALEPOCHS)
TestRMSEresults= MeanAcrossEpochT(TestRMSEresults,NUM_EPOCHS,NUM_FOLDS,RUNS,TOTALEPOCHS)



x = createList(0,NUM_EPOCHS-1)


plt.ylabel("RMSE", fontsize=14)
plt.xlabel("over epochs", fontsize=14)
plt.errorbar(x=x,y=TrainRMSEresults,yerr=atrstdRMSE ,color='r', label = 'train')
plt.errorbar(x=x,y=TestRMSEresults,yerr=atestdRMSE ,color='b', label = 'test')

plt.legend(loc='lower right')
plt.grid(True,color='k')
plt.show() 




plt.ylabel("Loss", fontsize=14)
plt.xlabel("over epochs", fontsize=14)

plt.errorbar(x=x,y=TrainLossResults,yerr=atrstdLoss ,color='r',label = 'train')
plt.errorbar(x=x,y=TestLossResults,yerr=atestdLoss,color='b',label = 'test')
plt.legend(loc='lower right')
plt.grid(True,color='k')
plt.show()


print(TrainRMSEresults)
print('/n')
print(TestRMSEresults)
print('/n')
print(atrstdRMSE)
print('/n')
print(atestdRMSE)
print('/n')
print('/n')
print(TrainLossResults)
print('/n')
print(TestLossResults)
print('/n')
print(atrstdLoss)
print('/n')
print(atestdLoss)
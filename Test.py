import pandas as pds
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import keras

##imports desired data
dataframeX = pds.read_csv('KaggleV2-May-2016.csv', usecols=[1,2,5,7,8,9,10,11,12,13])
dataframeY = pds.read_csv('KaggleV2-May-2016.csv')


##makes data usable numerical values
def genderToInt(gender):
    if gender == 'M': 
        return 0
    else:
        return 1

def ifShowToInt(noShow):
    if noShow == 'No':
        return 0
    else:
        return 1

##applys the test to numerical value filters 
dataframeY.NoShow = dataframeY.NoShow.apply(ifShowToInt)
dataframeX.Gender = dataframeX.Gender.apply(genderToInt)


##debug code
print(dataframeX.head())
print(dataframeY.head())

##random number generator
seed = 7
np.random.seed(seed)

##declears layers
model = Sequential()
model.add(Dense(11, input_shape=(10,), init = 'uniform', activation='sigmoid'))#Sigmoid function applied to output  to make variations more palletable.
model.add(Dense(11, init='uniform', activation = 'sigmoid'))
model.add(Dense(14, init='uniform', activation='sigmoid'))
model.summary()

tbCallback = keras.callbacks.TensorBoard(log_dir='/tmp/keras_logs', write_graph=True)

model.compile(loss='mean_squared_error', optimizer='adam', metrics = ['accuracy'])
###trained here with 9 passes, and 30% splitoff for validation
model.fit(dataframeX.values, dataframeY.values, epochs=9, batch_size=50, verbose=1, validation_split=0.3, callbacks=[tbCallback])
import pandas as pds
import numpy as np

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

##applys \the test to numerical value filters 
dataframeX.NoShow = dataframeX.NoShow.apply(ifShowToInt)
dataframeX.Gender = dataframeX.Gender.apply(genderToInt)


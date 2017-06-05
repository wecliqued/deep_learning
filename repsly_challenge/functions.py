from decimal import Decimal
from datetime import datetime

def class_replacement(classArray):
    newArray = []
    for i in classArray:
        if len(newArray) == 0:
            newArray.append(i)
        else:
            exists = 0
            for j in newArray:
                if i == j:
                    exists=1
            if exists == 0:
                newArray.append(i)
    numColumns = len(newArray)
    map = {}
    for x in range(numColumns):
        val = 10**x
        map[newArray[x]] = val
    return map

def date_replacement(dateArray):
    newArray = []
    firstDate = 0
    for i in dateArray:
        if len(newArray) == 0:
            newArray.append(i)
            firstDate = datetime.strptime(i, "%Y-%m-%d")
        else:
            exists = 0
            for j in newArray:
                if i == j:
                    exists=1
            if exists == 0:
                newArray.append(i)
    numColumns = len(newArray)
    map = {}
    for x in range(numColumns):
        val = datetime.strptime(newArray[x], "%Y-%m-%d") - firstDate
        map[newArray[x]] = val.days
    return map
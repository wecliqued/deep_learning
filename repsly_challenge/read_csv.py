# hello
import csv
import numpy
from Functions import *
import zipfile
#from io import StringIO

idCollection = []
boughtCollection = []
dateCollection = []
classCollection = []
moneyCollection = []

with open('/home/developer/Desktop/data.csv') as csvfile:
    mycsv = csv.reader(csvfile)

    csvfile.readline() #skip the 1st line
    for row in mycsv:
            idCollection.append(row[0])
            boughtCollection.append(row[1])
            dateCollection.append(row[2])
            classCollection.append(row[3])
            moneyCollection.append(row[4])

'''
print(idCollection)
print(boughtCollection)
print(dateCollection)
print(classCollection)
'''

classMap = class_replacement(classCollection)
dataMap = date_replacement(dateCollection)


#filehandle = open('/home/developer/Desktop', 'rb')
#zfile = zipfile.ZipFile(filehandle)
#data = StringIO.StringIO() #don't forget this line!
#reader = csv.reader(data)


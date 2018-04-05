import numpy as np
import csv
import random

def data_handle(file_name):
    file = open(file_name, 'r') #PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
    dataset = []
    embarked_buf = {'S': 0, 'C': 1 ,'Q': 2}
    for row in csv.DictReader(file):
        sex = 0
        if row['Sex'] == 'male':
            sex = 1
        age = float(-999)
        if row['Age']:
            age = float(row['Age'])
        embarked = -1
        if row['Embarked']:
            embarked = embarked_buf[row['Embarked']]
        fare = -999
        if row['Fare']:
            fare = float(row['Fare'])
        survived = 1
        if file_name == 'train.csv':
            if int(row['Survived']) == 0:
                survived = -1
            dataset.append([(float(row['Pclass']), sex, age, int(row['SibSp']), int(row['Parch']), fare, embarked ), survived])
        else:
            dataset.append([(float(row['Pclass']), sex, age, int(row['SibSp']), int(row['Parch']), fare, embarked ), int(row['PassengerId'])])
        #print(row['PassengerId']," | ", dataset[len(dataset) - 1])
    file.close()
    return dataset


def check_error(w, dataset):
    result = None
    error = 0
    for x, s in dataset:
        x = np.array(x)
        if int(np.sign(w.T.dot(x))) != s:
            result =  x, s
            error += 1
    print("error=%s/%s" % (error, len(dataset)))
    return result

def pla(dataset):
    w = np.zeros(7)
    w_buf = w
    max_t = 5000
    time = 0
    while check_error(w, dataset) is not None:
        x, s = check_error(w, dataset)
        w += s * x
        if w_buf.all() != w.all():
            w_buf = w
            time = 0
        else:
            time += 1
        if time == max_t:
            break
    return w

def test(w, data_test):
    result = []
    for x, s in data_test:
        x = np.array(x)
        if int(np.sign(w.T.dot(x))) == -1:
            result.append([s , 0])
        else:
            result.append([s , 1])
    return result

def writeToCsv(result):
    f = open('result.csv', 'w', newline='')
    w = csv.writer(f)
    w.writerows([['PassengerId','Survived']])
    w.writerows(result)
    f.close()
    print("Write complete!!")
    return

dataset = data_handle('train.csv')
w = pla(dataset)
print("result: " , w)
data_test = data_handle('test.csv')
result = test(w, data_test)
writeToCsv(result)

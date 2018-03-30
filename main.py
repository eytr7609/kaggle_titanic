import numpy as np
import csv
import random

def data_handle():
    file = open('train.csv', 'r') #PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
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
        dataset.append([(float(row['Pclass']), sex, age, int(row['SibSp']), int(row['Parch']), float(row['Fare']), embarked ), int(row['Survived'])])
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

#PLA演算法實作

def pla(dataset):
    w = np.zeros(7)
    w_buf = w
    max_t = 100
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

def pocket_pla(datas, limit):  
    ###############
    def _calc_false(vec):
        res = 0
        for data in datas:
            t = np.dot(vec, data[0])
            if np.sign(data[1]) != np.sign(t):
                res += 1
        return res
    ###############
    w = np.random.rand(7)
    least_false = _calc_false(w)
    res = w

    for i in range(limit):
        data = random.choice(datas)
        t = np.dot(w, data[0])
        if np.sign(data[1]) != np.sign(t):
            t = w + data[1] * data[0]
            t_false = _calc_false(t)

            w = t

            if t_false <= least_false:
                least_false = t_false
                res = t
    return res, least_false
#執行
dataset = data_handle()
w = pocket_pla(dataset, 100)
print(w)

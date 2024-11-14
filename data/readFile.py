import numpy as np

def readwine(str,i):
    if(i==0):
     file = open("data/artificial" + "/"+str+".arff", "r")
    else:
        file = open("data/real-world" + "/" + str + ".arff", "r")

    y = []
    label = []
    for line in file:
        # print(line)
        if (line.startswith("@") or line.startswith("%") or len(line.strip()) == 0):
            u = 0
        else:
            j = line.split(",")
            alpha = 1
            x=0
            if(not j[2].startswith("noise")):
              ak = float(j[2])
              if(ak == 6 or ak == 8):
                  alpha = 1
                  #x=0
            # print("line",line)

            k = []
            for i in range(len(j)-1):

             k.append(float(j[i+1])*alpha+x)
             #k.append(float(j[1])*alpha)
            label.append(int(j[0]))
            #if(not j[2].startswith("noise")):
            #    label.append(-1.00)
            y.append(k)
    return  np.array(y),np.array(label).reshape(1,len(label))[0]

def read_file(str,i):
    dic = {}
    classmember = 0
    if (i == 0):
        file = open("data/artificial" + "/" + str + ".arff", "r")
    elif i==1:
        file = open("data/real-world" + "/" + str + ".arff", "r")
    else:
        #ucipp-master/uci/abalone-3class.arff
        file = open("data/ucipp-master/uci/" + str + ".arff", "r")
    y = []
    label = []
    for line in file:
        # print(line)
        if (line.startswith("@") or line.startswith("%") or len(line.strip()) == 0):
            pass
        else:
            j = line.split(",")
            alpha = 1
            if ("?" in j):
                continue
            x = 0
            k = []

            for i in range(len(j) - 1):
                k.append(float(j[i]) * alpha + x)
            if (not j[len(j) - 1].startswith("noise")):
                str = j[len(j) - 1].rstrip()
                if(str in dic.keys()):
                    label.append(dic[str])
                else:
                    dic[str]= classmember
                    label.append(dic[str])
                    classmember +=1
                    ppppp = 9
            else:
                label.append(-1)
            y.append(k)
    return ((np.array(y), np.array(label).reshape(1, len(label))[0]))


def read_wdbc(str, i):
    dic = {}
    classmember = 0
    if (i == 0):
        file = open("artificial" + "/" + str + ".arff", "r")
    else:
        file = open("real-world" + "/" + str + ".arff", "r")
    y = []
    label = []
    for line in file:
        if (line.startswith("@") or line.startswith("%") or len(line.strip()) == 0):
            pass
        else:
            j = line.split(",")
            alpha = 1
            if ("?" in j):
                continue
            class_str = j[1]
            j = j[2:] + [class_str]
            x = 0
            k = []

            for i in range(len(j) - 1):
                k.append(float(j[i]) * alpha + x)
            if (not j[len(j) - 1].startswith("noise")):
                str = j[len(j) - 1]
                if (str in dic.keys()):
                    label.append(dic[str])
                else:
                    dic[str] = classmember
                    label.append(dic[str])
                    classmember += 1
            else:
                label.append(-1)
            y.append(k)
    # iter(zip(mnist_data, mnist_label))
    return iter(zip(np.array(y), np.array(label).reshape(1, len(label))[0]))

print(read_file('diamond9', 0))
#read_wdbc('wdbc', 1)
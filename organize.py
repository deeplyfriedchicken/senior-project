from sys import argv
import os

def organizeData(param):
    if not os.path.exists(param):
        os.makedirs(param)
    with open(argv[1], 'r') as fp:
        cnt = 1
        line = fp.readline()
        while line:
            data = line.split('\t')
            if (len(data) == 3):
                if (data[1] == param):
                    path = param + "/data{}.txt".format(data[0])
                    fh = open(path, "w")
                    fh.write(data[2])
                    fh.close()
            line = fp.readline()
            cnt += 1


organizeData('negative')
organizeData('positive')
organizeData('neutral')
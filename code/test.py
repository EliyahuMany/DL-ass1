import numpy as np

if __name__ == '__main__':

    W = np.zeros((3, 4))
    b = np.zeros(4)

    temp = np.shape(W)[1]
    print "temp"

    temp = [11,1,22,2,33,3,44,4]


    la = temp[0:-2:2]
    le = temp[1:-1:2]

    for w,b in zip(temp[0:-2:2], temp[1:-1:2]):
        print w,b

    for i, (W_temp, b_temp) in enumerate(zip(temp[-2::-2], temp[-1::-2])):
        print i,W_temp,b_temp
import numpy as np
import random
import matplotlib.pyplot as plt


R=np.matrix([
            [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1000],
            [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
            [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
            [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
            [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
            [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
            [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
            [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
            [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
            [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
            [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
            [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
            [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
            [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
            [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
            [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
            [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
            [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
            [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
            [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1] ])

# Q matrix
# Mapping all 20*20 cells to 400 * 5 successive  cells
Q = np.matrix(np.zeros([400,5]))


# Gamma parameter
gamma = 0.8



def convertToRow(i,j):
    return (i*20) + j

def convertToOriginal(rowIndex):
    i=int(rowIndex/20)
    j=rowIndex%20

    return  i,j


def move(dir,currentState):
    i, j = convertToOriginal(currentState)
    oldI =i
    olDJ =j
    if j==9 and  0<=i<=17 and dir==3: # wall , remain on current estate
        return i,j
    else:
        if dir==0:
            j=j-1
        elif dir==1:
            i=i-1
        elif dir==2:
            i=i+1
        elif dir==3:
            j=j+1

    if 0 <= i < 19 and 0 <= j <= 19:
        return i,j
    else:
        return oldI,olDJ

# 0:left, 1: up, :2 down, 3:right and 4:no move
def selectNextSA(currentState,moveDir):

    newI,newJ=move(moveDir,currentState)

    return convertToRow(newI,newJ)

goalState = convertToRow(0, 19)

def setAbsorbingState():
    Q[goalState,0]=1000
    Q[goalState,1]=1000
    Q[goalState,2]=1000
    Q[goalState,3]=1000
    Q[goalState,4]=1000

setAbsorbingState()

def training(maxEps):
    stepsArr = []
    episode = 0
    while episode<maxEps:
        initialState = convertToRow(0,0)
        currentState=initialState
        goal=False
        steps=0
        while not goal:
            if random.random()<=0.85:
                moveDir = np.argmax(Q[currentState,])
                nextState=selectNextSA(currentState,moveDir)
            else:
                moveDir=random.randint(0,3)
                i,j=move(moveDir,currentState)
                nextState=convertToRow(i,j)
            currI,currJ=convertToOriginal(currentState)
            # update Q (Q-Learning Formula)

            Q[currentState,moveDir]=R[currI,currJ] + (gamma * np.max(Q[nextState,]))
            currentState=nextState
            if currentState==goalState:
                goal=True

            steps+=1
        episode+=1
        print("episode # " + str(episode))
        print("     steps # " + str(steps))
        stepsArr.append(steps)
    print("     min steps # " + str(min(stepsArr)))
    return stepsArr


def findPath():
    goal=False
    currentState=0
    path=[]
    while not goal:
        path.append(convertToOriginal(currentState))
        moveDir = np.argmax(Q[currentState,])
        nextState=selectNextSA(currentState,moveDir)
        currI,currJ=convertToOriginal(currentState)
        # update Q (Q-Learning Formula)
        currentState=nextState
        if currentState==goalState:
            goal=True
            path.append(convertToOriginal(currentState))

    return path


maxEps=200
steps=training(maxEps)
path=findPath()


# Print selected sequence of steps
print("Selected path:")
print(path)
print("len : " +str(len(path)))
x=list(range(0,maxEps))
y=steps

plt.plot(x, y)
plt.show()



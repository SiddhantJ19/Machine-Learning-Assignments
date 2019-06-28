import matplotlib.pyplot as plt     

def plotgraph(x, y):
    plt.plot(x, y)
    
    plt.xlabel('Number of Hidden layer neurons')
    plt.ylabel('Accuracy')

    plt.show()

if __name__ == "__main__":
    plotgraph([1,2,3], [10,20,30])
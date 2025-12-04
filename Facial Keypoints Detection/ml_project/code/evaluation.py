import matplotlib.pyplot as plt

def plot_sample(x,y):
    """
    x:(96,96)
    y:(30,1)

    """
    y=y.reshape(15,2)
    fig,axis=plt.subplots(1,1,figsize=(12,9))
    axis.imshow(x,cmap='grey')
    axis.scatter(y[:,0],y[:,1],marker="x",s=50,color='red')
    return fig
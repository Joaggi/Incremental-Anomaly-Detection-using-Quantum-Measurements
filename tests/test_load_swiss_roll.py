import pytest 
from neuraldensityestimation.load_swiss_roll import load_swiss_roll 
import matplotlib.pylab as plt

def test_load_swiss_roll():
    X_train, X_train_density, X_test, X_test_density = load_swiss_roll(1000, 1000, 2)
    plt.axes(frameon = 0)
    plt.grid()
    plt.scatter(X_test[:,0],  X_test[:,1], c = X_test_density , alpha = .2, s = 3, linewidths= 0.0000001)
    plt.colorbar()
    plt.title('swiss_roll dataset')
    plt.savefig('reports/swiss_roll_dataset.png',dpi = 300)
    plt.show()

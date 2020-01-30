#Librerias

import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.manifold import Isomap

#Cargamos las imagenes de los digitos

digits = load_digits()

#Ponemos las subplots 10x10
fig, axes = plt.subplots(10,10,figsize=(8,8), subplot_kw ={'xticks':[],'yticks':[]},gridspec_kw=dict(hspace=0.1, wspace=0.1))

for i,ax in enumerate(axes.flat):
    ax.imshow(digits.images[i],cmap='binary',interpolation='nearest')
    ax.text(0.05,0.05, str(digits.target[i]),
        transform=ax.transAxes,color='green')

plt.show()

iso = Isomap(n_components=2)
iso.fit(digits.data)
data_projected = iso.transform(digits.data)
data_projected.shape

plt.scatter(data_projected[:, 0], data_projected[:, 1], c=digits.target,
edgecolor='none', alpha=0.5,
cmap=plt.cm.get_cmap('rainbow', 10))
plt.colorbar(label='digit label', ticks=range(10))
plt.clim(-0.5, 9.5)

plt.show()
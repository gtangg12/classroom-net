import pickle
import matplotlib.pyplot as plt

with open('train_losses.pkl', 'rb') as f:
    train_losses = pickle.load(f)

dl, dm, totloss = list(zip(*train_losses))
n_samples = list(range(0, 6*len(train_losses), 6))
plt.plot(n_samples, dl, label='3d distillation loss')
plt.plot(n_samples, dm, label='segmentation dist\' loss')
plt.plot(n_samples, totloss, label='total loss')

plt.legend()
plt.savefig('trainlosses.png')


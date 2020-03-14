import torch
from torch.autograd import Variable
import numpy as np
from auto_encoder import autoencoder,load_dataset
from sklearn.cluster import KMeans

model = autoencoder()
states = torch.load('conv_autoencoder.pth')
model.load_state_dict(states['state_dict'])

result = {}
for batch_idx, (data, labels, paths) in enumerate(load_dataset()):
    img = data[:,0,:,:].unsqueeze(1)
    img = Variable(img).cuda()
    enc, output = model(img)
    result[paths] = enc


kmeans = KMeans(n_clusters=7, n_init=20, n_jobs=4)
y_pred_kmeans = kmeans.fit_predict(result.values())
print(y_pred_kmeans)

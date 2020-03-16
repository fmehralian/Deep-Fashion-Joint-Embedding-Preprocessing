from sklearn.cluster import KMeans
import json

with open('data/dict.json') as f:
  data = json.load(f)
kmeans = KMeans(n_clusters=20, n_init=20, n_jobs=4)
y_pred_kmeans = kmeans.fit_predict(list(data.values()))
idx = 0
for key in data.keys():
    print(str(key)+":"+str(y_pred_kmeans[idx]))
    idx += 1

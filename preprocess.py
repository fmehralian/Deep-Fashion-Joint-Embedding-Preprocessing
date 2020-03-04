import json
import pickle
import numpy as np
import clean
import os

version = 1


class Data:
	def __init__(self):
		with open("data/contents_v{}.txt".format(version)) as f:
			data = [line.rstrip() for line in f]
		self.data = data[1:]


class Output:
	def __init__(self):
		self.data = ([], [])
		self.sanity()

	def add(self, caption, vecs):
		for idx, vec in enumerate(vecs):
			if len(vec) > 0:
				self.data[0].append(caption)
				self.data[1].append(vec) 
		self.sanity()

	def sanity(self):
		if len(self.data[0]) != len(self.data[1]):
			print("Data is not same length!")

	def save(self, name=""):		
		# Captions		
		captions = ""
		for cap in self.data[0]:
			captions += cap + "\n"
		if not os.path.exists('preprocessed_data'):
			os.makedirs('preprocessed_data')
		open("preprocessed_data/icons_v{}_caps.txt".format(version), "w").write(captions.rstrip())

		# Vectors 
		vecs = np.array(self.data[1])		
		np.save('preprocessed_data/icons_v{}_ims.npy'.format(version), vecs)

		# Final sanity check
		print("Difference (should be zero): ", (len(captions.split("\n"))-1) - len(vecs))
		return

# Load data, output, and image features
data = Data()
output = Output()
image_vecs = pickle.load(open("image_feature_dictionary.pkl", "rb"))

idx = 0
for caption in data:
    idx += 1
    output.add(clean.caption(caption.split(",")[-1]), [image_vecs[str(idx)]])

# Save our output
output.save()

# Sanity check
output.sanity()

# Done.
print("Script done.")

import torch
from sklearn.cluster import KMeans
from torch import nn
from torch.autograd import Variable
from torch.nn.modules.flatten import Flatten
from torchvision import transforms, datasets
from torchvision.utils import save_image
import os

version = 1
train = True
if torch.cuda.is_available():
    device=torch.device("cuda")
else:
    device = torch.device("cpu")
if not os.path.exists('dc_img'):
    os.mkdir('dc_img')


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 244, 244)
    return x


num_epochs = 100
batch_size = 128
learning_rate = 1e-3


img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
path_to_images = "data/bw_icons_v{}/".format(version)


class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


def load_dataset():
    data_path = 'data/bw_icons_v1/'
    train_dataset = ImageFolderWithPaths(
        root=data_path,
        transform=img_transform #torchvision.transforms.ToTensor()
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64,
        num_workers=0,
        shuffle=True
    )
    return train_loader

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 82, 82
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 41, 41
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 21, 21
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1),  # b, 8, 20, 20
            nn.Conv2d(8, 1, 3, stride=1, padding=1), # 1, 20, 20
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1, 8, 2, stride=2),  # b, 8, 20, 20
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 16, 5, stride=3),  # b, 16, 62, 62
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1),  # b, 8, 123, 123
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 244, 244
            nn.Tanh()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


model = autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                             weight_decay=1e-5)
if train:
    for epoch in range(num_epochs):
        for batch_idx, (data, labels, paths) in enumerate(load_dataset()):
            img = data[:,0,:,:].unsqueeze(1)
            if torch.cuda.is_available():
                img = Variable(img).to(device)
            # ===================forward=====================
            enc, output = model(img)
            loss = criterion(output, img)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'
              .format(epoch+1, num_epochs, loss.data.item()))
        if epoch % 10 == 0:
            pic = to_img(output.cpu().data)
            save_image(pic, 'dc_img/image_{}.png'.format(epoch))

    torch.save(model.state_dict(), 'conv_autoencoder.pth')
else:

    model = autoencoder()
    states = torch.load('conv_autoencoder.pth', map_location=device)
    model.load_state_dict(states)

    result = {}
    for batch_idx, (data, labels, paths) in enumerate(load_dataset()):
        img = data[:, 0, :, :].unsqueeze(1)
        img = Variable(img).to(device)
        enc, output = model(img)
        enc = enc.view(enc.size(0), -1)
        for i in range(len(paths)):
            result[paths[i]] = enc[i].tolist()

    kmeans = KMeans(n_clusters=7, n_init=20, n_jobs=4)
    y_pred_kmeans = kmeans.fit_predict(result.values())
    print(y_pred_kmeans)

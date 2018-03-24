# Train scent model.
# - Kenta Iwasaki

import numpy as np
import pretrainedmodels
import pretrainedmodels.utils as utils
import torch
import torch.nn.functional as F
from torch.utils import data
from sklearn.metrics import r2_score
from torch import nn, optim
from torch.optim import lr_scheduler


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.relu = nn.ReLU()
        self.avg_pool = nn.AvgPool2d(7, stride=1, padding=0)
        self.dropout = nn.Dropout()
        self.last_linear = nn.Linear(1056, 10)

    def forward(self, x):
        x = F.relu(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.last_linear(x)
        return x


features_model = pretrainedmodels.__dict__['nasnetamobile'](num_classes=1000, pretrained='imagenet')
features_model.eval()

classifier_model = Classifier()

features_model.cuda()
classifier_model.cuda()

criterion = nn.MultiLabelSoftMarginLoss()

load_image = utils.LoadImage()
transform_image = utils.TransformImage(features_model)

images = []

for image_index in range(1, 13):
    image = load_image('data/%d.jpg' % image_index)
    image = transform_image(image)

    images.append(image)
images = torch.stack(images, dim=0)

labels = np.genfromtxt('scents.csv', delimiter='\t', skip_header=1)[:12, 1:]
for index in range(len(labels)):
    labels[index] = labels[index] / max(labels[index])
labels = torch.FloatTensor(labels)

dataset = data.TensorDataset(images, labels)
loader = data.DataLoader(dataset, batch_size=4)

optimizer = optim.SGD(classifier_model.parameters(), lr=0.1, momentum=0.9)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

class_labels = "citrus	floral	fruity	woody	oriental	musk	aromatic	water	mossy	green".split(
    "\t")

# Train.

classifier_model.train()

for epoch in range(150):
    total_loss = 0
    for (batch_images, batch_labels) in loader:
        batch_images, batch_labels = torch.autograd.Variable(batch_images.cuda()), torch.autograd.Variable(batch_labels.cuda())
        outputs = classifier_model(features_model.features(batch_images))

        loss = criterion(outputs, batch_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.cpu().data[0]

    total_loss /= images.size()[0]
    print("[Epoch %d] Loss: %.8f" % (epoch, total_loss))

    scheduler.step(total_loss)

# Test.

classifier_model.eval()

outputs = classifier_model(features_model.features(torch.autograd.Variable(images.cuda())))
for predicted_label in outputs:
    label_probs, label_indices = torch.topk(predicted_label, 3)
    label_probs = label_probs.cpu().data.numpy()
    label_indices = label_indices.cpu().data.numpy()

    for (nth_prob, nth_index) in zip(label_probs, label_indices):
        print("%.4f: %s" % (nth_prob, class_labels[nth_index]))
    print()

outputs = outputs.cpu().data.numpy()
labels = labels.cpu().numpy()

for i in range(len(outputs)):
    outputs[i] = outputs[i] + abs(min(outputs[i]))
    outputs[i] = outputs[i] / max(outputs[i])

print(outputs)
print(labels)

score = r2_score(labels, outputs)
print(score)

torch.save(classifier_model.state_dict(), 'scent_model.pt')

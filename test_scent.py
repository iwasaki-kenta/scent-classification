# Test scent model.
# - Kenta Iwasaki

import numpy as np
import pretrainedmodels
import pretrainedmodels.utils as utils
import torch
import torch.nn.functional as F
from torch import nn


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
features_model.cuda()

classifier_model = Classifier()
classifier_model.load_state_dict(torch.load('scent_model.pt'))
classifier_model.eval()
classifier_model.cuda()

load_image = utils.LoadImage()
transform_image = utils.TransformImage(features_model)

class_labels = "citrus	floral	fruity	woody	oriental	musk	aromatic	water	mossy	green".split(
    "\t")

image = load_image('test/Screen Shot 2018-03-13 at 7.56.47 PM.png')
image = transform_image(image)

image = torch.autograd.Variable(image.unsqueeze(0).cuda())

outputs = classifier_model(features_model.features(image))
for predicted_label in outputs:
    label_probs, label_indices = torch.topk(predicted_label, 3)
    label_probs = label_probs.cpu().data.numpy()
    label_indices = label_indices.cpu().data.numpy()

    for (nth_prob, nth_index) in zip(label_probs, label_indices):
        print("%.4f: %s" % (nth_prob, class_labels[nth_index]))
    print()

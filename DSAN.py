import torch
import torch.nn as nn
import lmmd
from vit import vitB16

class DSAN(nn.Module):

    def __init__(self, num_classes=31):
        super(DSAN, self).__init__()
        self.vit_encoder = vitB16()
        self.lmmd_loss = lmmd.LMMD_loss(class_num=num_classes)

        # couche de classification
        self.cls_fc = nn.Linear(1000, num_classes)

    # forward method (fonction default d'entrainement en torch)
    def forward(self, source, target, s_label):
        source = self.vit_encoder(source)

        s_pred = self.cls_fc(source)
        target = self.vit_encoder(target)

        t_label = self.cls_fc(target)
        # calcule la perte d'adaptation
        loss_lmmd = self.lmmd_loss.get_loss(source, target, s_label, torch.nn.functional.softmax(t_label, dim=1))
        return s_pred, loss_lmmd

    #fonction de prédiction
    def predict(self, x):
        x = self.vit_encoder(x)
        return self.cls_fc(x)

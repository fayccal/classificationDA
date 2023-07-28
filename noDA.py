import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

train_dir = 'data/whole/myrsnadata'
test_dir = 'data/whole/myguan'
nclass = 2
size = 224

# transformation appliqué sur les données de training
transform = transforms.Compose(
        [transforms.Resize([256, 256]),
         transforms.RandomCrop(224),
         transforms.RandomRotation(10),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize((0.425, 0.415, 0.405), (0.205, 0.205, 0.205))
        ])

# transformation appliqué sur les données de test
transform_tests = torchvision.transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.425, 0.415, 0.405), (0.255, 0.245, 0.235))
    ])

# récupération des données et création dataloaders
train_data = torchvision.datasets.ImageFolder(root=train_dir,transform=transform)
test_data = torchvision.datasets.ImageFolder(root=test_dir,transform=transform_tests)

train_loader = DataLoader(train_data,batch_size=32,shuffle=True, num_workers=2)
test_loader= DataLoader(test_data,batch_size=32,shuffle=True,num_workers=2)

# check disponibilité gpu
train_on_gpu = torch.cuda.is_available()
device =  torch.device('cuda' if torch.cuda.is_available else 'cpu')

# utilise un modele de classification resnet50
model = torchvision.models.resnet50(pretrained=True)
for param in model.parameters():
    param.required_grad = False

num_ftrt = model.fc.in_features
model.fc = nn.Linear(num_ftrt, nclass)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4,6], gamma=0.06)
epochs = 10

for epoch in range(1,epochs+1):
    train_loss = 0.0
    valid_loss = 0.0

    # mode entrainement
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # déplacement des tensor sur le gpu
        if train_on_gpu:
            data, target = data.to(device), target.to(device)
        # clear les gradients
        optimizer.zero_grad()
        output = model(data)
        # calcule batch loss
        loss = criterion(output, target)
        # backward pass
        loss.backward()
        optimizer.step()

        # update train loss
        train_loss += loss.item()*data.size(0)

    model.eval()
    train_loss = train_loss/len(train_loader.sampler)
    # évaluation de la perte a la fin de l'epoch
    scheduler.step()
    print('Epoch: {} \t Training Loss: {:.3f}'.format(epoch, train_loss))






size = 224
correct_count, all_count = 0,0
#détermine l'accuracy sur les data de tests
for images, labels in test_loader:
    for i in range(len(labels)):
        # passage sur le gpu si disponible
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
        # extraction de l'image
        img = images[i].view(1,3,size,size)
        with torch.no_grad():
            logps = model(img)

        #convertion proba
        ps = torch.exp(logps)
        probab = list(ps.cpu()[0])
        pred_label = probab.index(max(probab))
        true_label = labels.cpu()[i]
        #check si les 2 labels match
        if(true_label == pred_label):
            correct_count += 1
        all_count += 1

print("\n Model Accuracy=",(correct_count/all_count)*100)
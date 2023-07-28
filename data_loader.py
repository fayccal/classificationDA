from torchvision import datasets, transforms
import torch
import os
from torch.utils.data import WeightedRandomSampler

# retourne un dataloader sur les images utilisé pour l'entrainement au quel on été appliqué les transformations
def load_training(root_path, dir, batch_size, ctransfo, kwargs):
    if ctransfo == False:
        transform = transforms.Compose(
            [transforms.Resize([256, 256]),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ])
        data = datasets.ImageFolder(root=os.path.join(root_path, dir), transform=transform)
        train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    else:
        # les transformations changé seront ajouté bientot

        transform = transforms.Compose(
            [transforms.Resize([256, 256]),
            transforms.RandomCrop(224),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
            ])
        data = datasets.ImageFolder(root=os.path.join(root_path, dir), transform=transform)
        weights = make_weights_for_balanced_classes(data.imgs, len(data.classes))
        weights = torch.DoubleTensor(weights)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
        train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, sampler=sampler, drop_last=True, **kwargs)
    return train_loader

# retourne un dataloader sur les images utilisé pour la phase de test au quel on été appliqué les transformations
def load_testing(root_path, dir, batch_size, kwargs):
    transform = transforms.Compose(
        [transforms.Resize([224, 224]),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
         ])
    data = datasets.ImageFolder(root=os.path.join(root_path, dir), transform=transform)
    test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, **kwargs)
    return test_loader





def make_weights_for_balanced_classes(images, nclasses):                        
    count = [0] * nclasses  
    #compte l'occurence de chaque classe                                                    
    for item in images:                                                         
        count[item[1]] += 1         
    #calcule les poids pour chaque classes                                            
    weight_per_class = [0.] * nclasses                                      
    N = float(sum(count))                                                   
    for i in range(nclasses):                                                   
        weight_per_class[i] = N/float(count[i])                                 
    weight = [0] * len(images)                 
    #liste de poids pour chaque classes                             
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]                                  
    return weight 

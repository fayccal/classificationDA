import torch
from torchvision import transforms
from PIL import Image
import os
from DSAN import DSAN


# génere un dictionnaire de toute les classes pour etre utilisé durant la prédiction
def generer_dict_class(chemin_repertoire):
    correspondance_classes = {}
    classes = sorted(os.listdir(chemin_repertoire))
    for i, classe in enumerate(classes):
        if os.path.isdir(os.path.join(chemin_repertoire, classe)):
            correspondance_classes[i] = classe
    return correspondance_classes



import argparse

def main():
    #argument à donner au programme: le modele a utilisé l'image a classifié, et le data_path pour la prédiction
    parser = argparse.ArgumentParser(description="Description de votre script.")
    parser.add_argument('modele', type=str, help='modèle utilisé')
    parser.add_argument('image', type=str, help='image a classifié')
    parser.add_argument('data_path', type=str, help='chemin des données pour utilisé une chaine de caractère')

    args = parser.parse_args()

    #chemin_repertoire_classes = 'data/OFFICE31/webcam/'
    chemin_repertoire_classes = args.data_path
    correspondance_classes = generer_dict_class(chemin_repertoire_classes)

    # Chemin du fichier contenant le modèle pickle
    chemin_image = args.image
    image = Image.open(chemin_image)

    chemin_model = args.modele

    # load le modèle torch sauvegardé
    modele = torch.load(chemin_model)

    # Transformation de l'image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Redimensionner l'image à la taille attendue par le modèle
        transforms.ToTensor(),  # Convertir l'image en tenseur
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normaliser les valeurs des pixels
    ])
    image = transform(image)

    modele.eval()

    image = image.unsqueeze(0).cuda()  # Ajouter une dimension pour représenter le batch (1 image) et passer au gpu
    output = modele.predict(image)

    # Traiter les prédictions
    _, predicted_class = torch.max(output, 1)
    predicted_class_id = predicted_class.item()
    predicted_class_name = correspondance_classes[predicted_class_id]
    print("Classe prédite :", predicted_class_name)
    
if __name__ == "__main__":
    main()

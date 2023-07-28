import cv2
import os
import pandas as pd
import pydicom
from PIL import Image

# remove tout les image de mauvaise qualité (trop de noire nous permet de les détecter)
def clean_lowq(dataset_path):
    # Seuil de pourcentage pour considérer une image comme principalement noire
    threshold_black_ratio = 0.2
    for filename in os.listdir(dataset_path):
        # traite les images jpeg
        if filename.endswith(".jpeg"):
            image_path = os.path.join(dataset_path, filename)
            image = cv2.imread(image_path)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Calculez le pourcentage de pixels noirs dans l'image
            total_pixels = gray_image.shape[0] * gray_image.shape[1]
            black_pixels = total_pixels - cv2.countNonZero(gray_image)
            black_ratio = black_pixels / total_pixels

            # Vérifiez si l'image est principalement noire ou a une petite taille
            if black_ratio >= threshold_black_ratio:
                os.remove(image_path)

# resize tout nos images
def resize_all(input_folder, output_folder):
    # Parcours des fichiers dans le dossier d'entrée
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpeg"):
            # Chemin complet du fichier d'entrée
            input_path = os.path.join(input_folder, filename)

            # Chemin complet du fichier de sortie
            output_path = os.path.join(output_folder, filename)

            # Chargement de l'image
            image = Image.open(input_path)

            # Redimensionnement de l'image en 224x224 pixels
            resized_image = image.resize((224, 224))

            resized_image.save(output_path)

            print("L'image '{}' a été redimensionnée.".format(filename))

# convertion des dicom
def convert_dicom_to_jpg(dicom_path, size=(224, 224)):
    # Lecture du fichier DICOM
    ds = pydicom.dcmread(dicom_path)

    # Extraction de l'image pixel
    pixel_array = ds.pixel_array

    # Conversion en image PIL
    image = Image.fromarray(pixel_array)

    resized_image = image.resize(size)

    return resized_image

# map sur tout le csv de rsna
def convert_dicom(datafolder,outfolder, Normalfolder, Pneumoniafolder):

    df = pd.read_csv('stage_2_train_labels.csv')
    df = df.drop(['x', 'y', 'width', 'height'], axis=1)

    for i in range(len(df)):
        pid = df.iloc[i]['patientId']
        value = df.iloc[i]['Target']
        wholepath = datafolder + '/' + pid + '.dcm'
        img = convert_dicom_to_jpg(wholepath)
        if value == 0:
            img.save(outfolder + Normalfolder + "/" + pid + '.jpeg' , 'JPEG')
        else:
            img.save(outfolder + Pneumoniafolder + "/" + pid + '.jpeg', 'JPEG')


# applique clahe au dataset
def clahe_all(input_folder, output_folder):
    # parcours des fichiers dans le dossier d'entrée
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpeg"):
            # chemin complet du fichier d'entrée
            input_path = os.path.join(input_folder, filename)

            # chemin complet du fichier de sortie
            output_path = os.path.join(output_folder, filename)

            # ouvre l'image
            image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
            # application de clahe
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            clahe_image = clahe.apply(image)

            cv2.imwrite(output_path, clahe_image)

            print("L'image '{}' a été convertie.".format(filename))
# Overview

Ceci est un modèle pytorch réalisant de la classification d'image entre domaine grace à l'utilisation de domain adaptation, reprenant DSAN et les Vit.

## datasets

Voici les liens de téléchargement vers les datasets, qui doivent etre déposé dans un folder créée au nom de 'data'.

- office31: https://drive.google.com/file/d/0B4IapRTv9pJ1WGZVd1VDMmhwdlE/view?resourcekey=0-gNMHVtZfRAyO_t2_WrOunA
- officehome: https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view?resourcekey=0-2SNWq0CDAuWOBRRBL7ZZsw
- CXR data: https://drive.google.com/file/d/1aPTLskr0ROrP47INrOOSlnUdQTCHSUau/view?usp=drive_link
- seg model: https://drive.google.com/file/d/1e-EJn-f4d7-2Vnf6jg3bC4LpZqpkbnwZ/view?usp=sharing
## Utilisation

Tout d'abord il est nécessaire de télécharger les poids pré-entrainé sur imagenet ici et déposé ce fichier dans le répertoire courant:
https://drive.google.com/file/d/1FJaS80TWwS_9Z3hNIAsKF4nKXALGUk88/view?usp=drive_link

Pour lancé l'entrainement avec main.py après avoir installer les dépendances, plusieurs arguments peuvent etre utilisé voici les suivants:

- root_path qui est décrit le chemin vers le dataset voulu.
- src : nom du répertoire étant le source domain
- tar : nom du répertoire étant le target domain
- nclass : nombre de classe présente
- ctransfo : pour utilisé les transformation custom pour les données CXR.(coming)

les autres paramètres sont de préférence à ne pas toucher.
les datasets doivent etre déplacé dans le répertoire data.
le meilleur modèle est gardé à la fin de chaque epoch.

un simple python3 main.py lance le programme avec la config par défaut, la config lancé peut etre modifié en rajoutant --OptionName Valeur pour chaque valeur voulant etre changé.


Pour lancé le serving.py il est nécessaire de fournir les arguments suivants:

- modele : le modèle voulant etre utilisé (format pth, pt)
- image : l'image voulant etre classifié
- data_path : le chemin vers les données, pour pouvoir display une chaine de charactère et non un simple ID durant la prédiction.

```
python3 serving path/to/modele.pth path/to/image.jpg path/to/data
```



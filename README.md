# kaggle_mnist

Code python 3 servant à la reconnaissance d'image de la distribution MNIST modifiée. Le but du projet est d'implémenter un modèle d'apprentissage automatique étant capable de reconnaître la somme de deux chiffres sur une image.
Les résultats sur l'ensemble de test sont enregistrés dans un fichier .csv au même endroit où a été exécuté le fichier main.py ou main_DL.py. Des graphiques de performances sur la fonction de coût de la précision sont aussi sauvegardés.

## Librairies nécessaires
  - Pytorch 1.11+
  - Numpy 1.23+
  - Pandas 1.5+
  
## Exécution
### main.py
code pour la régression logistique
  - python3 main.py [OPTION]...
 
### main_DL.py
code pour les rseaux de neurones pondérés
  -  python3 main.py [OPTION]..
  
## options possibles
main.py
  - --batch_size <int>  (gère la taille d'un lot, defaut: 16)
  - --max_epoch <int>  (gère le nombre d'époques d'entraînement, défaut: 64) 
  - --decay <float>  (régularisation Ridge, défaut: 1e-5)
  - --lr <float>    (rythme d'entraînement, défaut: 1e-3)
  - --train_pct <float>  (proportion des données pour l'entraînement, défaut: 0.7)
  - --val_train_pct <float>  (proportion des données pour la validation, défaut: 0.15)
  - --path_data <path>  (chemin relatif ou absolu où se trouve les données, défaul: "mnist/")

main_DL
  - --batch_size <int>  (gère la taille d'un lot, defaut: 128)
  - --max_epoch <int>  (gère le nombre d'époques d'entraînement, défaut: 1024) 
  - --decay <float>  (régularisation Ridge, défaut: 1e-5)
  - --lr <float>    (rythme d'entraînement, défaut: 1e-3)
  - --train_pct <float>  (proportion des données pour l'entraînement, défaut: 0.7)
  - --val_train_pct <float>  (proportion des données pour la validation, défaut: 0.15)
  - --path_data <path>  (chemin relatif ou absolu où se trouve les données, défaul: "mnist/")
  - --use_scheduler (lorsqu'appelé, cette option permet d'utiliser un planificateur pour les modèles)

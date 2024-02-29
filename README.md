# Taï National Park Camera Trap Image Classification Challenge

## Problem Description
In this challenge, participants are tasked with classifying species in camera trap images from Taï National Park. Images capture seven species types plus images with no animals. The goal is to assist conservation efforts by accurately predicting species presence in these images.

## Data
- **Training and Testing Sets**: Images in `train_features` and `test_features` directories.
- **Metadata**: `train_features.csv` and `test_features.csv` include image ID, filepath, and site.

## Features
- Images come with additional attributes: `id`, `filepath`, and `site`.
- There's no site overlap between training and testing data, emphasizing model generalization.

## Labels
- Eight possible labels: seven species and 'blank' for no animals.
- Each image is labeled for one species group or as blank.

## Submission and Evaluation
- Submit probabilities for each of the eight classes for images in `test_features`.
- Evaluation is based on log loss, with lower values indicating better performance.

## Tips
- Consider different environments by ensuring train/test sets have disjoint sites.
- Use image augmentation to address variations within images.

Good luck, and for more information or questions, visit the user forum for this competition.

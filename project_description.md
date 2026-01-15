# Hand-gesture recognition v1

## Requirements

1. Write a script to automatically create a dataset that includes at least two different static hand gestures.

2. Train a Neural Network (Example: CNN) on your dataset so that it can classify which gesture is visible in the image.

3. Apply data augmentation to the dataset and test how much it improves the network. Report on the benefits of each augmentation.
   - Examples: Geometric Transformations, Color Transformations, Blurring and Sharpening
   - Test at least 4 different augmentations and find the best combination

4. Test how the dataset size affects the performance and how it changes with data augmentation.

5. Report on the confusion matrix and accuracy for each class.

## Dataset Specifications

- The dataset should contain labeled images with:
  - First gesture (class 1)
  - Second gesture (class 2)
  - Without a hand gesture (class 3)

- Split your dataset into 80% training data and 20% test data to evaluate the accuracy of your model.

- The dataset should have between 100-1000 images per class.

## Evaluation

- Compare the performance of the network with the F1-score.
- Report confusion matrix and accuracy for each class.

## Technical Constraints

- For the network, you can use models that are not pre-trained or pre-trained on other tasks.

- Implement the augmentation functions yourself (no framework like torchvision but you can use basic functions from OpenCV).

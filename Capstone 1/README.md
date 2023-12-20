The dataset for this project downloaded from  https://www.kaggle.com/datasets/puneet6060/intel-image-classification

This is image data of Natural Scenes around the world.

There are 6 different classes in this dataset

- buildings
- forest
- glacier
- mountain
- sea
- street

With my EDA you can see that there are almost equal distribution by classes in train dataset, so it won't make any problems with it. And this dataset contains images with same resolution 150x150 so there won't be any changes in models.

I'v tried 3 differnet Tensorflow models for this project

1. Simple CNN model 

2. Same simple CNN model with augmentation

3. Transfer learning

In my opinion the best solution was simple CNN model with the best accuracy 81% on test dataset

by launching predict.py you can download image and get class for this image

With this project you can classify images of natural Scenes 

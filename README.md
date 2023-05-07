# Brats21
BraTS21 Segmentation challenge 

To train: 

python3 run.py -T --PATH ./data 



To evaluate: 

python3 run.py -E --PATH ./data G



To predict (ground truth files not needed):

python3 run.py -P --PATH ./data

Note that you can replace "data" with "smalldata" for a quicker test, although it may not work for 
Trainign since the train test split may not have enough data. 


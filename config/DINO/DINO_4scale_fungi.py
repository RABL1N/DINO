_base_ = ['DINO_4scale.py']

num_classes = 2
dn_labelbook_size = 3

# Matching MaskDINO hyper-parameters
lr = 1e-6
lr_backbone = 1e-7
batch_size = 4
epochs = 1000
lr_drop = 1001 # No decay during this run

# Matching MaskDINO's image resolution
data_aug_scales = [1024]
data_aug_max_size = 1024

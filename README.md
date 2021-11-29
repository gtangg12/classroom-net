# classroom-net

## datalake
build will populate the datalake with waymo data (rgb, range, camera transformations) \
build will call pretrained models \
dataset will define dataset class and allow downstream to query for respective data

## teachers
pretrained model implementations \
each impl should provide function that takes in data and outputs result

## main framework
### experiments
ablation, generalization, teach eval, etc...

### student
main impl of classroom-net framework (see design image in group chat) \
don't worry about stuff in red shape in design image (precompute) since dataset will deal with that \
impl dataloader(dataset) which produces ((x, x_1, x_2...), y) where x_i are aux inputs, x input rgb image, y depth map

### utils
put util functions here e.g. func that compute recall and precision

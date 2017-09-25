from img_cap_data import ImgCapData

#train data
train_data = ImgCapData(basedir='/Users/lyfpcy/ml/aichallenge/train/', anno_file='/caption.json', max_sample = 5000)

train_data.build_all_and_save()
train_data.extract_feature()

# val data
val_data = ImgCapData(basedir='/Users/lyfpcy/ml/aichallenge/val/', anno_file='/caption.json', max_sample = 1000)

val_data.build_all_and_save()
val_data.extract_feature()
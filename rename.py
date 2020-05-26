import os

dir = '/home/mksnkv/Documents/classification/class/data/train/dogs/'
error_counter = 0
images = os.listdir(dir)

i = 1
for filename in images:
    older = dir + filename
    new_name = "dogs" + str(i) + ".jpg"
    newer = dir + new_name
    os.rename(older, newer)
    i += 1

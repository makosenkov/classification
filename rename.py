import os

dir = '/home/user/Документы/convnets/raw-img/cows/'
error_counter = 0
images = os.listdir(dir)

i = 1
for filename in images:
    older = dir + filename
    new_name = "cow" + str(i) + ".jpg"
    newer = dir + new_name
    os.rename(older, newer)
    i += 1

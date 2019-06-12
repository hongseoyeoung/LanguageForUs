import os


for i in range(0, 58):
    for j in range(1201,2401):
        path = "gestures/" + str(i) + "/" + str(j) + ".jpg"
        if os.path.isfile(path):
            os.remove(path)

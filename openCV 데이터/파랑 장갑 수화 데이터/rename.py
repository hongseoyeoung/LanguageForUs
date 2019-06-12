import os

for i in range(0, 58):
    files = os.listdir("gestures_plus2/" + str(i) + "/.")
    path = "gestures_plus2/" + str(i) + "/"
    for file in files:
        word = file.split('.')
        new = str(int(word[0]) + 200) + ".jpg"
        os.rename(path+file, path+new)
        

import pandas as pd
import random

path = "clients_1.txt"

data = pd.read_csv(path, sep = ' ')
data.columns = ['Timestamp', 'File_ID', "File_Size"]
DataLength = len(data)

lastTime = max(data["Timestamp"])

print(data)
for i in range(15):
    val =  random.randint(1, 50)
    for j in range(len(data)):
        if data["File_ID"][j] == val:
            data["Timestamp"][j] = random.randint(0, lastTime)

data.sort_values("Timestamp", inplace=True)
data.reset_index(drop=True, inplace=True)
print(data)
f = open("clients_1.txt", "w")
for i in range(len(data)):
    f.write(str(data['Timestamp'][i]))
    f.write(" ")
    f.write(str(data['File_ID'][i]))
    f.write(" ")
    f.write(str(data['File_Size'][i]))
    f.write("\n")
f.close()

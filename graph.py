import pandas as pd
import matplotlib.pyplot as plt

# read the CSV file into a pandas DataFrame
df = pd.read_csv("D:\SR\statistics\srf_4_train_results.csv")

cols = ["Epoch", "Loss_D", "Loss_G", "PSNR", "SSIM"]
data = df[cols]

# set the x axis to be column_1
x = data["Epoch"]

# plot each column as a separate line graph
for col in cols[1:]:
    y = data[col]
    plt.figure()
    plt.plot(x, y)
    plt.xlabel("Epoch")
    plt.ylabel(col)
    plt.title(col + " vs Epoch")

# display the graphs
plt.show()
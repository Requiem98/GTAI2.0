from libraries import *
import baseFunctions as bf

data = pd.read_csv("./Data/gta_data/path_11/data.csv", sep=",")

data["steering_angle"].hist(bins=80)
data["throttle"].hist(bins=80)
data["brake"].hist(bins=80)
data["speed"].hist(bins=80)


data["steering_angle"][(data["steering_angle"]!=0.5)].hist(bins=80)
data["brake"][(data["brake"]!=0)].hist(bins=80)


data["steering_angle"].plot()
data["brake"].plot()


image = io.imread(data.iloc[0,4])
plt.imshow(image)





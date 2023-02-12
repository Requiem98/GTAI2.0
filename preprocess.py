from libraries import *
import baseFunctions as bf

#=============================== remove frames ============================
"""
data = pd.read_csv("./Data/gta_data/path_traffic_4/data.csv", sep=",")


idx = np.concatenate([data.index[2741:2900], data.index[24220:]])

data.drop(idx, inplace=True)

data.drop(data.index[14326:], inplace=True)

data.to_csv("./Data/gta_data/path_traffic_4/data.csv", index=False)
"""

#=============================== normal dataset ============================




data1 = pd.read_csv("./Data/gta_data/path_1/data.csv", sep=",")
data2 = pd.read_csv("./Data/gta_data/path_2/data.csv", sep=",")
data3 = pd.read_csv("./Data/gta_data/path_3/data.csv", sep=",")
data4 = pd.read_csv("./Data/gta_data/path_4/data.csv", sep=",")
data5 = pd.read_csv("./Data/gta_data/path_5/data.csv", sep=",")
data6 = pd.read_csv("./Data/gta_data/path_6/data.csv", sep=",")
data7 = pd.read_csv("./Data/gta_data/path_7/data.csv", sep=",")
data8 = pd.read_csv("./Data/gta_data/path_8/data.csv", sep=",")
data9 = pd.read_csv("./Data/gta_data/path_9/data.csv", sep=",")
data10 = pd.read_csv("./Data/gta_data/path_10/data.csv", sep=",")
data11 = pd.read_csv("./Data/gta_data/path_11/data.csv", sep=",")


data_tot_norm = pd.concat([data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11], ignore_index=True)

data_tot_norm["steering_angle"].hist(bins=80)

new_data = bf.cap_hist(data_tot_norm, k=7000)

new_data.to_csv("./Data/gta_data/data_tot_norm.csv")


#=============================== Traffic dataset ============================

data1 = pd.read_csv("./Data/gta_data/path_traffic_1/data.csv", sep=",")
data2 = pd.read_csv("./Data/gta_data/path_traffic_2/data.csv", sep=",")
data3 = pd.read_csv("./Data/gta_data/path_traffic_3/data.csv", sep=",")
data4 = pd.read_csv("./Data/gta_data/path_traffic_4/data.csv", sep=",")
data5 = pd.read_csv("./Data/gta_data/path_traffic_5/data.csv", sep=",")
data6 = pd.read_csv("./Data/gta_data/path_traffic_6/data.csv", sep=",")

data_tot_traffic = pd.concat([data1, data2, data3, data4, data5, data6], ignore_index=True)

data_tot_traffic["steering_angle"].hist(bins=80)

new_data = bf.cap_hist(data_tot_traffic, k=10000)

new_data.to_csv("./Data/gta_data/data_tot_traffic.csv")

#=============================== Traffic + normal dataset =====================
data_tot_norm = pd.read_csv("./Data/gta_data/data_tot_norm.csv", index_col=0)
data_tot_traffic = pd.read_csv("./Data/gta_data/data_tot_traffic.csv", index_col=0)

data_tot = pd.concat([data_tot_norm, data_tot_traffic], ignore_index=True)

data_tot["steering_angle"].hist(bins=80)
data_tot["throttle"].hist(bins=80)
data_tot["speed"].hist(bins=80)


data_tot.to_csv("./Data/gta_data/data_tot.csv")

#=============================== Test dataset =====================

data_test = pd.read_csv("./Data/gta_data/path_TEST/data.csv", sep=",")

data_test["steering_angle"].hist(bins=80)
data_test["steering_angle"].plot()

data_test.to_csv("./Data/gta_data/data_TEST.csv")

#=============================== Sampled dataset ==============================
data = pd.read_csv("./Data/gta_data/data_tot.csv", index_col=0)
data = data.sample(frac=1/3)

data.to_csv("./Data/gta_data/data_tot_sampled.csv")

#=============================== Segmentation dataset ============================


data = pd.read_csv("C:/Users/amede/Downloads/segmentation dataset/segment_data.csv", index_col=0)

data_train, data_test = bf.create_train_test_dataframe(data, group_n=1, 
                                                       test_size=0.2, 
                                                       save_dir = "C:/Users/amede/Downloads/segmentation dataset/", 
                                                       test_file_name = "segment_data_test.csv", train_file_name = "segment_data_train.csv", save=True)

#===============================================================================

data_tot = pd.read_csv(DATA_ROOT_DIR + 'imagesTOT_balanced.csv', index_col=0)


plt.hist(normalize_steering(data_tot["steeringAngle"]), bins=80)

plt.hist(data_tot["acceleration"], bins=80)


plt.hist(data_tot["steeringAngle"] , bins=80)

train_dataset = bf.GTADataset2("imagesTOT_balanced.csv", DATA_ROOT_DIR, bf.preprocess, mmap=True)

train_dataset.__getitem__(0)

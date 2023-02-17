from libraries import *
import baseFunctions as bf


plt.style.use('dark_background')

figure, ax = plt.subplots(1,1, figsize=(10,5))

ax.plot(sa_pred[1:], label = "Prediction")
ax.plot(sa_gt[1:], alpha=0.8, label = "Ground truth")
ax.grid(linestyle = "--", alpha = 0.5)
ax.legend()
ax.set_xlabel("Frames")
ax.set_ylabel("Steering Angle")

plt.savefig("./Data/plots/Dnet_st.png")



#=============================================


data_tot["steering_angle"].hist(bins=80, figsize= (10,5), color = "gold")
plt.grid(linestyle = "--", alpha = 0.5)
plt.xlabel("Steering Angle")
plt.ylabel("Samples")

plt.savefig("./Data/plots/data_distr2.png")



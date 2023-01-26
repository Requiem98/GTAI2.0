from libraries import *
import baseFunctions as bf
from torchvision.models import vit_h_14, ViT_H_14_Weights


#==============================================================================
#================================ GET IMAGE ===================================
#==============================================================================

data = pd.read_csv("./Data/gta_data/data_TEST.csv")

image = io.imread(data["path"][0])

plt.imshow(image)

image = image[200:480, :]
image = F.to_pil_image(image)
image = F.resize(image, (224,224))
image

mmap = image[500:580, 50:130]

plt.imshow(mmap)


#preprocessing of mmap
mmap = F.to_pil_image(mmap)
mmap = F.resize(mmap, (40,40))
mmap = F.to_tensor(mmap)

#==============================================================================
#================================ GET BATCH ===================================
#==============================================================================

test_dataset = bf.GTADataset("data_TEST.csv", DATA_ROOT_DIR, mmap=True)

test_dl = DataLoader(test_dataset, 
                        batch_size=1, 
                        num_workers=0)

for batch in test_dl:
    break


batch["img"].shape

batch["mmap"].shape


conv_map = MMAP_CONV()

conv_map(batch["mmap"]).shape



weights = ViT_B_16_Weights.DEFAULT
model = vit_b_16(weights=weights)

train_nodes, eval_nodes = get_graph_node_names(model)

backbone = create_feature_extractor(model, train_return_nodes =train_nodes[-3:-2], eval_return_nodes = eval_nodes[-3:-2]) #bx192x768

backbone(batch["img"])["encoder.ln"][:, 1:].shape

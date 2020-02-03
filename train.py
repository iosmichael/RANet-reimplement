from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import yaml
from matplotlib import pyplot as plt
from dataloader import DAVIS2017Dataset

config = yaml.load(open('config.yaml', 'r'), Loader=yaml.SafeLoader)
train_config = config['train']
# writer = SummaryWriter(train_config['tensorboard_directory'])
# writer add_scalar('train/loss', loss, num_of_iter)

dataset = DAVIS2017Dataset(config, transforms=None)
dataloader = DataLoader(dataset)
print(dataloader)

for imgs, masks, sizes in dataloader:
    m, img = masks[0], imgs[0]
    # masked image based on entity
    masked_img = np.expand_dims(np.array(m) > 0, axis=2) * np.array(img)
    plt.imshow(masked_img)
    plt.show()
    break

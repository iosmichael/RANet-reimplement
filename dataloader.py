import os, glob
import yaml
import cv2
import numpy as np
from PIL import Image
from collections import defaultdict
from matplotlib import pyplot as plt

class DAVIS2017Dataset(object):

    def __init__(self, config, transforms=None):
        self.config = config['dataset']
        self._load_trainval()
        self.transforms = transforms

    def _load_trainval(self):
        # load train categories
        train_fpath = os.path.join(self.config['directory'], self.config['train'])
        assert os.path.isfile(train_fpath)
        self.train_categories = [c.strip() for c in open(train_fpath, 'r')]
        print(f'load train categories: # {len(self.train_categories)}')
         # load train categories
        val_fpath = os.path.join(self.config['directory'], self.config['val'])
        assert os.path.isfile(val_fpath)
        self.val_categories = [c for c in open(val_fpath, 'r')]
        print(f'load validation categories: # {len(self.val_categories)}')
        self._load_image_mask()
    
    def _load_image_mask(self, is_train=True):
        assert len(self.train_categories) != 0
        assert len(self.val_categories) != 0
        # load train mask
        imgs = defaultdict(list)
        labels = defaultdict(list)
        if is_train:
            categories = self.train_categories
        else:
            categories = self.val_categories

        for c in categories:
            for img in glob.glob(os.path.join(self.config['directory'], self.config['image_directory'], c, '*.jpg'), recursive=True):
                imgs[c].append(img)
            for mask in glob.glob(os.path.join(self.config['directory'], self.config['label_directory'], c, '*.png'), recursive=True):
                labels[c].append(mask)
        self.dataset = {'imgs': imgs, 'labels': labels}

    def label2masks(self, label, objs):
        masks = []
        for obj in objs:
            masks.append(Image.fromarray(np.ones(label.shape) * (label == obj) * 255).convert('L'))
        return masks

    def label2mask(self, label, obj):
        return Image.fromarray(np.ones(label.shape) * (label == obj) * 255).convert('L')

    def masks2label(self, masks, objs):
        assert len(masks) == len(objs)
        label = torch.zeros(masks[0].size())
        for idx, msk in enumerate(masks):
            ids = torch.nonzero(msk)
            if len(ids) > 0:
                label[ids[:, 0], ids[:, 1], ids[:, 2]] = idx + 1
        return label

    def __len__(self):
        return len(self.train_categories)
            
    def __getitem__(self, index):
        '''
        input:
        - index: category id
        output:
        - return: a list of pictures and masks
        '''
        # need to set train/val mode
        c = self.train_categories[index]
        target = Image.open(self.dataset['labels'][c][0])
        objs_ids = list(set(np.array(target).reshape(-1)))
        objs_ids.remove(0)
        # random select on label
        np.random.shuffle(objs_ids)
        entity_id = objs_ids[0]
        # prepare for the output data
        imgs, masks, sizes = [], [], []
        for idx, fpath in enumerate(self.dataset['imgs'][c]):
            img = Image.open(fpath).convert('RGB')
            label = Image.open(self.dataset['labels'][c][idx])
            mask = self.label2mask(np.array(label).astype(np.float), entity_id) 
            imgs.append(img)
            masks.append(mask)
            sizes.append(list(img.size))
        return imgs, masks, sizes

if __name__ == '__main__':
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.SafeLoader)
    dataset = DAVIS2017Dataset(config)
    imgs, masks, sizes = dataset[20]
    # example demonstration
    m, img = masks[0], imgs[0]
    # masked image based on entity
    masked_img = np.expand_dims(np.array(m) > 0, axis=2) * np.array(img)
    plt.imshow(masked_img)
    plt.show()

class MiniImageNet(Dataset):
    def __init__(self, setname, args):
        
        data_base_dir = 'data/miniimagenet'
        if os.path.exists(data_base_dir):
            pass
        else:
            print ('Download miniImageNet from Google Drive.')
            os.makedirs(data_base_dir)
            os.system('sh scripts/download_miniimagenet.sh')

        IMAGE_PATH = 'data/miniimagenet/images'
        SPLIT_PATH = 'data/miniimagenet/split'

        csv_path = osp.join(SPLIT_PATH, setname + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        data = []
        label = []
        lb = -1
        self.wnids = []

        for l in lines:
            name, wnid = l.split(',')
            path = osp.join(IMAGE_PATH, name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            data.append(path)
            label.append(lb)

        self.data = data
        self.label = label
        self.num_class = len(set(label))

        if setname == 'val' or setname == 'test':
            image_size = 84
            self.transform = transforms.Compose([
                transforms.Resize([92, 92]),
                transforms.CenterCrop(image_size),

                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])
        elif setname == 'train':
            image_size = 84
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label
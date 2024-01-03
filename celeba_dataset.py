from torchvision.datasets import CelebA

class CelebAGender(CelebA):
    def __init__(self, root, split="train", transform=None,
                 target_transform=None, download=False):
        super(CelebAGender, self).__init__(root, split, "attr", transform, target_transform, download)

    def __getitem__(self, item):
        img, label = super(CelebAGender, self).__getitem__(item)
        code = label[20]
        return img, code


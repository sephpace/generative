
import os

from PIL import Image, ImageOps
from torch.utils.data import Dataset
import torchvision.transforms.functional as F

DATA_DIR = os.path.abspath('data')


class PokemonDataset(Dataset):
    """
    Pictures of 809 pokemon.
    """

    def __init__(self, add_mirrored=False):
        """
        Constructor.

        Args:
            add_mirrored (Optional[bool]): If True will double the dataset size with mirrored images. Default is False.
        """
        self.data = []
        for root, _, files in os.walk(os.path.join(DATA_DIR, 'pokemon/images/images')):
            for file in files:
                # Load image data
                name = os.path.splitext(file)[0]
                img = Image.open(os.path.join(root, file))
                img = img.convert('RGBA') if img.mode != 'RGBA' else img
                self.data.append((self.to_tensor(img), name))

                # Add mirrored images
                if add_mirrored:
                    m_img = ImageOps.mirror(img)
                    self.data.append((self.to_tensor(m_img), name))

    def __getitem__(self, idx): return self.data[idx]

    def __len__(self): return len(self.data)

    @staticmethod
    def to_tensor(img):
        """
        Converts the given image into a pytorch tensor.

        Args:
            img (PngImageFile): The image to convert.

        Returns:
            (Tensor): The tensor representation of the image.
        """
        t = F.to_tensor(img)
        error_msg = f'Wrong image size! {t.shape}, should be (4, 120, 120)!'
        assert t.shape == (4, 120, 120), error_msg
        return t

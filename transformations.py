import torchvision.transforms as transforms
import PIL
from PIL import Image
from skimage import transform
from datasets import TrainDatasetH5
import matplotlib.pyplot as plt
import os


def bicubic_upsampler(img, new_shape=(500, 500)):
    """
    To Do: Check if this works for all types of images (--> including PIL)!
    Check https://scikit-image.org/docs/dev/api/skimage.transform.html#skimage.transform.warp
    for more info on the function
    """
    # alternative for PIL new_image = image.resize((800,400),Image.BICUBIC)

    new_image = transform.resize(img, output_shape=new_shape, order=3)  # order=3 is bicubic
    return new_image


def nearest_upsampler(img, new_shape):
    new_image = transform.resize(img, output_shape=new_shape, order=0)  # order=0 is nearest neighbor
    return new_image


SIMPLE_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])])

MINIMALIST_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(p=0.5)
])

TRANSFORM_LR = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])])

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])])

if __name__ == '__main__':
    train_file_name = '91-image_x2.h5'
    TRAIN_FILE = os.path.abspath(os.path.join(__file__, f'../data_SR/91-Images/{train_file_name}'))
    dataset_h5 = TrainDatasetH5(TRAIN_FILE)

    image, label = dataset_h5.__getitem__(1)

    # plt.imshow(image[0, :, :], cmap='gray')
    # plt.show()

    # plt.imshow(label[0], cmap='gray')
    # plt.show()

    gt = label[0, :, :]
    pred = image[0, :, :]

    print(gt.shape)
    print(pred.shape)

    reshaped_pred_bicubic = bicubic_upsampler(pred, new_shape=(50, 50))
    reshaped_pred_nearest = nearest_upsampler(pred, new_shape=(50, 50))

    plt.imshow(reshaped_pred_nearest, cmap='gray')
    plt.show()

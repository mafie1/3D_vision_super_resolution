import torchvision.transforms as T
import PIL
from PIL import Image
from skimage import transform

def bicubic_upsampler(image, new_shape = (500, 500)):
    """To Do: Check if this works for images that are not necessarily PIL images!"""
    #new_image = image.resize((800,400),Image.BICUBIC)
    new_image = transform.resize(image, output_shape=new_shape, order = 3) #order=3 is bicubic
    return new_image

def nearest_upsampler(image, new_shape):
    new_image = transform.resize(image, output_shape=new_shape, order=0)  # order=0 is nearest neighbor
    return new_image


TRANSFORM_LR = T.Compose([
    T.Resize(256),
    T.CenterCrop(256),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])])


if __name__ == '__main__':
    from datasets import TrainDatasetH5
    import matplotlib.pyplot as plt

    TRAIN_FILE = "/Users/luisaneubauer/Documents/WS 2021:22/3D Reconstruction/super_resolution/data/Set5/91-image_x2.h5"
    dataset_h5 = TrainDatasetH5(TRAIN_FILE)

    image, label = dataset_h5.__getitem__(1)

    #plt.imshow(image[0, :, :], cmap='gray')
    #plt.show()

    #plt.imshow(label[0], cmap='gray')
    #plt.show()

    gt = label[0, :, :]
    pred = image[0, :, :]

    print(gt.shape)
    print(pred.shape)

    reshaped_pred_bicubic = bicubic_upsampler(pred, new_shape=(50, 50))
    reshaped_pred_nearest = nearest_upsampler(pred, new_shape=(50, 50))


    plt.imshow(reshaped_pred_nearest, cmap = 'gray')
    plt.show()



import torchvision.transforms as T
import PIL

def bicubic_upsampler(image):
    pass

def nearest_upsampler(image):
    pass


TRANSFORM_IMG = T.Compose([
    T.Resize(256),
    T.CenterCrop(256),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])])



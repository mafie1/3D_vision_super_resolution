3D_vision_super_resolution

##What you can do with this project:
- Train Neural Networks (SRCNN, VDSR, ResNet) for the task of super-resolution --> train.py
- This project file contains the custom datasets (--> datasets.py for BSD100, Set5, 91-Images)
- You can compare the results either during training (--> train.py) or with a separete script (--> compare_baselines-py)
- SSIM and PSNR are implemented for images as numpy arrays or as (batch) pytorch tensors.
- The Custom Loss (Charbonnier Loss) can be found in losses.py
- In case you want to apply data augmentation/transforms (including e.g. image normalization), these are to be found in transformations.py
- the utils.py file contains helful functions that make our life easier. These include a function to display tensors as images using matplotlib, setting random seeds or an Average Meter that we use during training to store the PSNR and SSIM values
- In train.py has tensorboard support --> by default commented out 
- We have specified the required packages in requirements.txt
- Use the demo.ipynb file to take a look at one of our trained models and proof of concept. 
- Due to hardware limitations, the model sizes are quite small and we would like to train for more epochs if we could get our hands on a proper GPU.
- The write-up can be found under Super_Resolution_Project_Docu.pdf


___
Our student IDs are:
- Luisa Nadine Neubauer (Physik B.A., 4013523)
- Mikail Deniz Cayoglu (Data and Computer Science, 3437201)
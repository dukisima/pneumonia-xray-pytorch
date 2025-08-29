from torchvision import transforms

######### Transforms for train (it includes augmentations) #######
#Resize the image so that they are uniform and from faster train time
#Totensor is to convert images from PIL image to a tensor
#Grayscale is to convert all the x-ray images to RGB bc ImageNet is expecting that
#Normalize
#Augmentations are so that the algorithm doesn't learn some random stuff about the image

train_transforms = transforms.Compose([transforms.Resize((224,224)),
                                       transforms.ToTensor(),
                                       transforms.Grayscale(num_output_channels=3),
                                       transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
                                       #Augmenttions
                                       transforms.RandomResizedCrop(size=224, scale=(0.8,1.0)),
                                       transforms.RandomHorizontalFlip(p=0.5)
                                       ])


######### Transforms for test and val (it does not includes augmentations) #######

test_val_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                          transforms.ToTensor(),
                                          transforms.Grayscale(num_output_channels=3),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                          ])



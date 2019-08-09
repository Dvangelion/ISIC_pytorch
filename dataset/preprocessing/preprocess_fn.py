from torchvision import transforms


def senet_preprocessing(is_training=True):

    train_transformation = transforms.Compose([
        transforms.Resize((300,300)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(),
        transforms.RandomAffine(degrees=30),
        transforms.RandomCrop((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.6337, 0.6060, 0.5936],
                            std=[0.1393, 0.1832, 0.1970])
    ])


    validation_transformation = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.6337, 0.6060, 0.5936],
                            std=[0.1393, 0.1832, 0.1970])
    ])

    if is_training == True:
        return train_transformation

    elif is_training == False:
        return validation_transformation




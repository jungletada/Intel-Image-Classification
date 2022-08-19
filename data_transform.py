from torchvision import transforms

""" data settings"""
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.RandomCrop([224, 224]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(45),
        transforms.RandomGrayscale(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),

    'test': transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


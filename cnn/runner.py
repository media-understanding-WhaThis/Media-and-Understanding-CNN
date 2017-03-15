import torch
import torchvision.transforms as transforms
from cnn.pytorch_main import CNN
from cnn.pytorch_main import Net
from cnn.pytorch_plant_dataset import PlantDataset

if __name__ == '__main__':
    batch_size = 4

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])

    train_set = PlantDataset(root='data/plantset', transform=transform, train=True)
    test_set = PlantDataset(root='data/plantset', transform=transform, train=False)

    cnn = CNN(Net())
    cnn.run_train_plant(train_set, test_set, batch_size)
    # cnn.run_test_plant(test_set, batch_size)

    saved_net = torch.load('data/trained_network.p')
    predicted_num, predicted_string = CNN.single_prediction(
        saved_net, 'data/plantset/sunflower/1001901836_3d592b5f93.jpg')
    # print(classes[predicted_num])

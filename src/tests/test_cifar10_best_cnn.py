import os

import torch
import torchvision
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

# =========================== CUDA =====================================================================================

torch.multiprocessing.set_sharing_strategy('file_system')

cuda_available = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda_available else "cpu")

print('torch version {}\ntorchvision version {}'.format(torch.__version__, torchvision.__version__))
print('CUDA available? {} Device is {}'.format(cuda_available, device))
if cuda_available:
    print('Number of available CUDA devices {}\n'.format(torch.cuda.device_count()))

dir_name = os.path.dirname(__file__)
# save_path = dir_name + '/../../generated_files/tensorboard_experiment'
save_path = dir_name + '/../../generated_files/experiment_8/model-0af70747-263c-4c78-8bac-fa32d2936c4c/'


# =========================== Hyper-params =============================================================================

max_num_of_epochs = 10
batch_size = 64
valid_size = 0.1


# =========================== Model def ================================================================================

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, padding=2)
        self.batch_nrm1 = torch.nn.BatchNorm2d(6)
        self.pool = torch.nn.MaxPool2d(2, 2)

        self.conv2 = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding=2)
        self.batch_nrm2 = torch.nn.BatchNorm2d(16)

        self.conv3 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        self.batch_nrm3 = torch.nn.BatchNorm2d(32)

        self.conv4 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.batch_nrm4 = torch.nn.BatchNorm2d(64)

        self.fc1 = torch.nn.Linear(64 * 4 * 4, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(self.batch_nrm1(torch.nn.functional.relu(self.conv1(x))))
        x = self.pool(self.batch_nrm2(torch.nn.functional.relu(self.conv2(x))))
        x = self.pool(self.batch_nrm3(torch.nn.functional.relu(self.conv3(x))))
        x = self.batch_nrm4(torch.nn.functional.relu(self.conv4(x)))

        x = x.view(-1, 64 * 4 * 4)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def write_model_summery(train_loader, _model):
    writer = SummaryWriter(save_path)
    images, labels = next(iter(train_loader))
    grid = torchvision.utils.make_grid(images)
    writer.add_image('images', grid)
    writer.add_graph(model=_model.cpu(), input_to_model=images, verbose=False)
    writer.close()
    if cuda_available:
        model.cuda()


def validate_model(epoch, max_num_of_epochs, prev_loss, valid_loader):
    val_accu, curr_loss = test_model(data_loader=valid_loader, validation_flag=True)
    print('Validation Epoch {}/{} stats ==== accuracy {:.4f} ==== loss {:.4f} ==== previous loss {:.4f}\n'.
          format(epoch + 1, max_num_of_epochs, val_accu, curr_loss, prev_loss))

    if curr_loss >= prev_loss:
        print('Early stopping criteria is meet, stopping training after {} epochs'.format(epoch + 1))
        return True, curr_loss, val_accu
    else:
        return False, curr_loss, val_accu


def test_model(data_loader, validation_flag=False):
    criterion = torch.nn.CrossEntropyLoss().cuda()
    correctly_labeled = 0
    total_predictions = 0
    running_loss = 0

    with torch.no_grad():
        for data in data_loader:

            images, labels = data
            if cuda_available:
                images = images.cuda()
                labels = labels.cuda()

            outputs = model(images)
            avg_batch_loss = criterion(outputs, labels)
            running_loss += avg_batch_loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correctly_labeled += (predicted == labels).sum().item()

    model_accuracy = correctly_labeled / total_predictions
    model_avg_loss = running_loss / len(data_loader)

    if validation_flag:
        return model_accuracy, model_avg_loss
    else:
        print('\nTest set stats === accuracy {:.4f} === loss {:.4f}\n'.format(model_accuracy, model_avg_loss))
        return model_accuracy


def train_model(train_loader, valid_loader):
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters())
    prev_valid_loss = float('inf')

    for epoch in range(max_num_of_epochs):  # loop over the dataset multiple times
        epoch_loss = 0
        epoch_correctly_labeled = 0
        epoch_total_labeled = 0

        print('Started training epoch {}/{}'.format(epoch + 1, max_num_of_epochs))

        for _iter, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            iter_per_epoch = len(train_loader)
            inputs, labels = data

            if cuda_available:
                inputs = inputs.cuda()
                labels = labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)

            avg_batch_loss = criterion(outputs, labels)
            avg_batch_loss.backward()
            optimizer.step()

            epoch_loss += avg_batch_loss.item()

            _, predicted = torch.max(outputs.data, 1)
            epoch_total_labeled += labels.size(0)
            batch_correctly_labeled = (predicted == labels).sum().item()
            epoch_correctly_labeled += batch_correctly_labeled

            # if _iter % 50 == 0:
            #     print('Epoch {} stats ==== Iteration {}/{} ==== Accuracy {} ===== loss {} ====='.
            #           format(epoch + 1, _iter, iter_per_epoch, batch_correctly_labeled / batch_size, loss))

        print('Training Epoch {}/{} stats ==== Accuracy {:.4f} ==== Avg Epoch Loss {:.4f} ===='.
              format(epoch + 1, max_num_of_epochs, epoch_correctly_labeled / epoch_total_labeled,
                     epoch_loss / len(train_loader)))

        _, prev_valid_loss, _ = validate_model(epoch, max_num_of_epochs, prev_valid_loss, valid_loader)


def save_pytorch_model():
    save_file_path = save_path + '/model_save.pt'
    torch.save(model, save_file_path)

    # torch.save({
    #     'model': model,
    #     'optimizer': optimizer,
    #     'epoch': epoch,
    #     'loss': loss
    # }, save_file)

    # torch.save(obj={
    #     'model_state_dict': model.state_dict(),
    #     'optimizer_state_dict': optimizer.state_dict(),
    #     'epoch': epoch,
    #     'loss': loss, }
    #     , f=save_file)


def load_pytorch_model():
    # load_file_path = save_path + '/model_save.pt'
    load_file_path = save_path + 'pytorch_model-0af70747-263c-4c78-8bac-fa32d2936c4c.pt'


    # model = torch.nn.Sequential()
    # model = torch.nn.DataParallel(model)

    model = torch.load(load_file_path)
    # model.load_state_dict(checkpoint['model_state_dict'])

    # optimizer = torch.optim.Adam(params=model.parameters())
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # epoch = checkpoint['epoch']
    # loss = checkpoint['loss']

    model.eval()
    # - or -
    # model.train()
    return model


# ======================================================================================================================

transform = torchvision.transforms.Compose([
    # torchvision.transforms.Resize(256),
    # transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

dir_name = os.path.dirname(__file__)
trainset = torchvision.datasets.CIFAR10(root=dir_name + '/../../data-sets/cifar10', train=True,
                                        download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root=dir_name + '/../../data-sets/cifar10', train=False,
                                       download=True, transform=transform)

num_train = len(trainset)
indices = list(range(num_train))
split = int(np.floor(valid_size * num_train))

np.random.seed()
np.random.shuffle(indices)
train_idx, valid_idx = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=train_sampler,
                                          num_workers=2)

validloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=valid_sampler,
                                          num_workers=2)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

# classes = ('plane', 'car', 'bird', 'cat',
#            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# model = Net().to(device)
# model = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=False).to(device)
# model = torchvision.models.alexnet(pretrained=False, progress=True).to(device)
# model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg11', pretrained=False).to(device)
# model = torchvision.models.vgg11(pretrained=False, progress=True).to(device)
# model = torch.hub.load('pytorch/vision:v0.6.0', 'densenet121', pretrained=False).to(device)
# model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=False).to(device)
# model = torch.hub.load('pytorch/vision:v0.6.0', 'inception_v3', pretrained=False).to(device)


# write_model_summery(trainloader, model)
# train_model(trainloader, validloader)
# test_model(testloader)
# save_pytorch_model()

model = load_pytorch_model()
test_model(testloader)

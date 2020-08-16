from abstract_models_generation import *
from pytorch_model_generation import *
from pytorch_model_train import *
import arch_gen_config as config


# ======================================================================================================================

def main():
    # TODO - these configurations are set for CIFAR-10 data-set, in the future make predefined set of configurations

    iterations = round(config.population_size)
    for model_num in range(iterations):
        model_id, abstract_model = generate_abstract_model()
        model = create_pytorch_model(abstract_model, model_id, apply_fix=True)
        pytorch_model = PytorchModel(model=model, model_id=model_id, model_num=model_num)
        model_test_accuracy, num_of_train_epochs = pytorch_model.set_train_and_test_model()
        save_abstract_model_to_csv(abstract_model, model_id, model_test_accuracy, num_of_train_epochs)


if __name__ == "__main__":
    main()



# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.batch_nrm1 = nn.BatchNorm2d(6)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.batch_nrm2 = nn.BatchNorm2d(16)
#         self.pool = nn.MaxPool2d(2, 2)
#
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)
#
#     def forward(self, x):
#         x = self.pool(self.batch_nrm1(F.relu(self.conv1(x))))
#         x = self.pool(self.batch_nrm2(F.relu(self.conv2(x))))
#
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
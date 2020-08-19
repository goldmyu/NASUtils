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
        model_test_accuracy, model_test_loss, num_of_train_epochs = pytorch_model.set_train_and_test_model()
        save_abstract_model_to_csv(abstract_model, model_id, model_test_accuracy, model_test_loss,num_of_train_epochs)


if __name__ == "__main__":
    main()


from config import init_configurations
from abstract_models_generation import *
from pytorch_model_generation import *
from pytorch_model_train import *


# ======================================================================================================================


def main():
    # TODO - these configurations are set for CIFAR-10 data-set, in the future make predefined set of configurations
    init_configurations(grid=False,
                        population_size=5000,
                        max_network_depth=20,
                        max_network_parallel_layers=1,
                        num_classes=10,
                        conv_max_height=20,
                        conv_max_width=20,
                        conv_max_filters=128,
                        conv_max_stride=1,
                        pool_max_height=2,
                        pool_max_width=2,
                        pool_max_stride=2,
                        dropout_max_rate=0.5,
                        linear_max_dim=10,
                        dataset_height=32,
                        dataset_width=32,
                        dataset_channels=3,
                        batch_size=256,
                        num_of_dataloader_workers=4,
                        max_num_of_epochs=20,
                        min_num_of_epochs=5,
                        validation_size=0.2,
                        logging_rate_initial=10,
                        models_save_path='../generated_files/experiment_6/',
                        log_weights=True,
                        log_activations=True)

    iterations = round(config['population_size'])
    for model_num in range(iterations):
        model_id, abstract_model = generate_abstract_model()
        model = create_pytorch_model(abstract_model, model_id, apply_fix=True)
        pytorch_model = PytorchModel(model=model, model_id=model_id, model_num=model_num)
        model_test_accuracy, num_of_train_epochs = pytorch_model.set_train_and_test_model()
        save_abstract_model_to_csv(abstract_model, model_id, model_test_accuracy, num_of_train_epochs)


if __name__ == "__main__":
    main()

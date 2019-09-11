from config import init_configurations
from abstract_models_generation import *
from pytorch_model_generation import *
from pytorch_model_train import *


# ======================================================================================================================

def main():
    # TODO - these configurations are set for CIFAR-10 data-set, in the future make predefined set of configurations
    init_configurations(grid=False,
                        population_size=100,
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
                        batch_size=128,
                        max_num_of_epochs=20,
                        min_num_of_epochs=5,
                        validation_size=0.2,
                        logging_rate_initial=200,
                        models_save_path='generated_files/experiment_1/')


    # population = initialize_population()
    for iter in range(config['population_size']):
        model_id, model = generate_abstract_model()
        pytorch_model = create_pytorch_model(model, model_id, apply_fix=True)
        set_train_and_test_model(pytorch_model, model_id)
        save_abstract_model_to_csv(model, model_id)


if __name__ == "__main__":
    main()


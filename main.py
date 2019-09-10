from config import init_configurations
from abstract_models_generation import *
from pytorch_model_generation import *
from pytorch_model_train import *


# ======================================================================================================================

def main():
    # TODO - these configurations are set for CIFAR-10 data-set, in the future make predefined set of configurations
    init_configurations(grid=False,
                        population_size=10000,
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
                        logging_rate_initial=25,
                        models_save_path='generated_files/experiment_1/')

    population = initialize_population()

    for model_tuple in population:
        model = finalize_model(model_tuple.get('model'))
        model_id = model_tuple.get('model_id')
        print('Creating pytorch model for model {}'.format(model_id))
        # print_model_structure(model)

        pytorch_model = create_pytorch_model(model, apply_fix=True)
        set_train_and_test_model(pytorch_model, model_id)
        # print(pytorch_model)


if __name__ == "__main__":
    main()


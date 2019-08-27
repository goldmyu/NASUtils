from config import init_configurations
from abstract_models_generation import *
from pytorch_model_generation import *
from pytorch_model_train import *


# ======================================================================================================================

def main():
    # TODO - these configurations are set for CIFAR-10 data-set, in the future make predefined set of configurations
    init_configurations(grid=False,
                        population_size=1000,
                        max_network_depth=10,
                        max_network_parallel_layers=1,
                        num_classes=10,
                        conv_max_height=20,
                        conv_max_width=20,
                        conv_max_channels=128,
                        conv_max_stride=1,
                        pool_max_height=2,
                        pool_max_width=2,
                        pool_max_stride=2,
                        dropout_max_rate=0.5,
                        linear_max_dim=10,
                        dataset_height=32,
                        dataset_width=32,
                        dataset_channels=3,
                        batch_size=64,
                        models_save_path='generated_files/')

    pop = initialize_population()
    model_tuple = pop[0]
    model = finalize_model(model_tuple.get('model'))
    print_model_structure(model)
    pytorch_model = create_pytorch_model(model, apply_fix=True)
    set_train_and_test_model(pytorch_model, model_tuple.get('model_id'))

    print(pytorch_model)


if __name__ == "__main__":
    main()


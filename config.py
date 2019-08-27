config = {}


def init_configurations(grid,
                        population_size,
                        max_network_depth,
                        max_network_parallel_layers,
                        num_classes,
                        conv_max_height,
                        conv_max_width,
                        conv_max_channels,#filters
                        conv_max_stride,
                        pool_max_height,
                        pool_max_width,
                        pool_max_stride,
                        dropout_max_rate,
                        linear_max_dim,
                        dataset_height,
                        dataset_width,
                        dataset_channels,
                        batch_size,
                        models_save_path):
    global config
    for key, value in locals().items():
        config[key] = value



# class PredefindConfigurations:

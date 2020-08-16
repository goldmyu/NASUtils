models_save_path = '/generated_files/experiment_7/'

grid = False
population_size = 5000

max_network_depth = 20
max_network_parallel_layers = 1
num_classes = 10

conv_max_height = 20
conv_max_width = 20
conv_max_filters = 128
conv_max_stride = 1

pool_max_height = 2
pool_max_width = 2
pool_max_stride = 2

dropout_max_rate = 0.5
linear_max_dim = 10

dataset_height = 32
dataset_width = 32
dataset_channels = 3

batch_size = 256

num_of_dataloader_workers = 4

max_num_of_epochs = 50
min_num_of_epochs = 50

validation_size = 0.2

logging_rate_initial = 10
log_only_at_epoch_end = True
log_weights = True
log_activations = True

import torch
from collections import defaultdict
from functools import reduce
import networkx as nx
import random
import models_generation
import numpy as np
from copy import deepcopy


def add_raw_to_weighted_population(pop, raw):
    for key, value in raw.items():
        pop[key] = value


def sum_evaluations_to_weighted_population(pop, evaluations, str_prefix=''):
    for metric, valuedict in evaluations.items():
        metric_str = metric
        if f"{str_prefix}train_{metric_str}" in pop:
            pop[f"{str_prefix}train_{metric_str}"] += valuedict['train']
            pop[f"{str_prefix}val_{metric_str}"] += valuedict['valid']
            pop[f"{str_prefix}test_{metric_str}"] += valuedict['test']


def get_model_state(model):
    if globals.get('cross_subject'):
        available_states = [x for x in model.keys() if 'model_state' in x]
        model_state_str = random.sample(available_states, 1)[0]
    else:
        model_state_str = 'model_state'
    return model[model_state_str]


def check_age(model):
    return globals.get('use_aging') and\
        random.random() < 1 - 1 / (model['age'] + 1)


def get_average_param(models, layer_type, attribute):
    attr_count = 0
    count = 0
    for model in models:
        layers = get_model_layers(model)
        for layer in layers:
            if isinstance(layer, layer_type):
                attr_count += getattr(layer, attribute)
                count += 1
    if count == 0:
        return 'NAN'
    return attr_count / count


def calculate_population_similarity(layer_collections, evolution_file, sim_count):
    sim = 0
    to_output = 3
    for i in range(sim_count):
        idxs = random.sample(range(len(layer_collections)), 2)
        score, output = models_generation.network_similarity(layer_collections[idxs[0]],
                                                             layer_collections[idxs[1]], return_output=True)
        sim += score
        if to_output > 0:
            with open(evolution_file, "a") as text_file:
                print(output, file=text_file)
            to_output -= 1
    return sim / sim_count


def calculate_one_similarity(layer_collection, other_layer_collections):
    sim = 0
    for other_layer_collection in other_layer_collections:
        score, output = models_generation.network_similarity(layer_collection,
                                                             other_layer_collection)
        sim += score
    return sim / len(other_layer_collections)


def inject_dropout(weighted_population):
    for pop in weighted_population:
        layer_collection = pop['model']
        for i in range(len(layer_collection)):
            if random.uniform(0, 1) < globals.get('dropout_injection_rate'):
                old_layer = layer_collection[i]
                layer_collection[i] = models_generation.DropoutLayer()
                if not models_generation.check_legal_model(layer_collection):
                    layer_collection[i] = old_layer
        pop['model_state'] = None


def remove_from_models_hash(model, model_set, genome_set):
    if globals.get('grid'):
        for layer in model.nodes.values():
            remove_layer = True
            for other_model in model_set:
                if not equal_grid_models(model, other_model):
                    for other_layer in other_model.nodes.values():
                        if layer['layer'] == other_layer['layer']:
                            remove_layer = False
                            break
                if not remove_layer:
                    break
            if remove_layer and layer['layer'] in genome_set:
                genome_set.remove(layer['layer'])
        if model in model_set:
            model_set.remove(model)
    else:
        for layer in model:
            remove_layer = True
            for other_model in model_set:
                if model != other_model:
                    for other_layer in other_model:
                        if layer == other_layer:
                            remove_layer = False
                            break
                if not remove_layer:
                    break
            if remove_layer and layer in genome_set:
                genome_set.remove(layer)
        if model in model_set:
            model_set.remove(model)


def get_model_layers(model):
    if globals.get('grid'):
        return [layer['layer'] for layer in model.nodes.values()]
    else:
        return model


def hash_model(model, model_set, genome_set):
    if globals.get('grid'):
        add_model = True
        for other_model in model_set:
            if equal_grid_models(model, other_model):
                add_model = False
        if add_model:
            model_set.append(model)
    else:
        if model not in model_set:
            model_set.append(model)
    layers = get_model_layers(model)
    for layer in layers:
        if layer not in genome_set:
            genome_set.append(layer)


def count_layer_type_in_pop(models, layer_type):
    count = 0
    for model in models:
        layers = get_model_layers(model)
        for layer in layers:
            if isinstance(layer, layer_type):
                count += 1
    return count


def num_of_models_with_skip_connection(weighted_population):
    total = 0
    for pop in weighted_population:
        if len(list(nx.all_simple_paths(pop['model'], 'input', 'output_conv'))) > 1:
            total += 1
    return total


def equal_grid_models(layer_grid_1, layer_grid_2):
    for i in range(layer_grid_1.graph['height']):
        for j in range(layer_grid_2.graph['width']):
            if layer_grid_1.nodes[(i,j)]['layer'] != layer_grid_2.nodes[(i,j)]['layer']:
                return False
    for edge in layer_grid_1.edges:
        if edge not in layer_grid_2.edges:
            return False
    for edge in layer_grid_2.edges:
        if edge not in layer_grid_1.edges:
            return False
    return True


def initialize_population():
    global config
    if config['grid']:
        model_init = models_generation.random_grid_model
    else:
        model_init = models_generation.random_model
    population = []
    for i in range(config['population_size']):
        new_rand_model = model_init()
        population.append(new_rand_model)
    return population


def cross_subject_shared_fitness(weighted_population):
    for item in weighted_population:
        fitness = item[f'val_{globals.get("ga_objective")}']
        subject_array = range(1, globals.get('num_subjects')+1)
        fitness_vector = [item[f'{i}_val_{globals.get("ga_objective")}'] for i in subject_array]
        denominator = 1
        dists = []
        for pop in weighted_population:
            pop_fitness = [pop[f'{i}_val_{globals.get("ga_objective")}'] for i in range(1, globals.get('num_subjects')+1)]
            dists.append(pdist([fitness_vector, pop_fitness])[0])
        dists_norm = [float(i)/max(dists) for i in dists]
        for dist in dists_norm:
            if dist < globals.get('min_dist'):
                denominator += 1-dist
        item['fitness'] = fitness / denominator


def normal_fitness(weighted_population):
    for pop in weighted_population:
        pop['fitness'] = pop[f'val_{globals.get("ga_objective")}']


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def calc_ensembles_fitness(ensembles, pop_fitnesses, weighted_population):
    for ensemble in ensembles:
        ensemble_fit = one_ensemble_fitness(weighted_population, ensemble)
        for pop in ensemble:
            pop_fitnesses[pop].append(ensemble_fit)
    return ensemble_fit


def ensemble_fitness(weighted_population):
    pop_fitnesses = defaultdict(list)
    for iteration in range(globals.get('ensemble_iterations')):
        pop_indices = list(range(globals.get('pop_size')))
        random.shuffle(pop_indices)
        ensembles = list(chunks(pop_indices, globals.get('ensemble_size')))
        calc_ensembles_fitness(ensembles, pop_fitnesses, weighted_population)
    for pop_fitness in pop_fitnesses.items():
        weighted_population[pop_fitness[0]]['fitness'] = np.average(pop_fitness[1])


def one_subject_one_ensemble_fitness(weighted_population, ensemble, str_prefix=''):
    ensemble_preds = np.mean([weighted_population[i][f'{str_prefix}val_raw'] for i in ensemble], axis=0)
    pred_labels = np.argmax(ensemble_preds, axis=1).squeeze()
    ensemble_targets = weighted_population[ensemble[0]][f'{str_prefix}val_target']
    ensemble_fit = getattr(utils, f'{globals.get("ga_objective")}_func')(pred_labels, ensemble_targets)
    return ensemble_fit


def one_ensemble_fitness(weighted_population, ensemble):
    if globals.get('cross_subject'):
        ensemble_fit = 0
        for subject in globals.get('subjects_to_check'):
            ensemble_fit += one_subject_one_ensemble_fitness(weighted_population, ensemble, str_prefix=f'{subject}_')
        return ensemble_fit / len(globals.get('subjects_to_check'))
    else:
        return one_subject_one_ensemble_fitness(weighted_population, ensemble)


def calculate_ensemble_fitness(weighted_population, ensemble):
    if globals.get('cross_subject'):
        ensemble_fit = 0
        for subject in range(1, globals.get('num_subjects') + 1):
            ensemble_fit += one_ensemble_fitness(weighted_population, ensemble)
        return ensemble_fit / globals.get('num_subjects')
    else:
        return one_ensemble_fitness(weighted_population, ensemble)


def permanent_ensemble_fitness(weighted_population):
    pop_indices = list(range(globals.get('pop_size')))
    ensembles = list(chunks(pop_indices, globals.get('ensemble_size')))
    perm_ensemble_fitnesses = []
    for i, ensemble in enumerate(ensembles):
        ensemble_fit = calculate_ensemble_fitness(weighted_population, ensemble)
        ensemble_fit_dict = {'group_id': i, 'fitness': ensemble_fit}
        perm_ensemble_fitnesses.append(ensemble_fit_dict)
        for pop_index in ensemble:
            weighted_population[pop_index]['fitness'] = ensemble_fit_dict['fitness']
            weighted_population[pop_index]['group_id'] = ensemble_fit_dict['group_id']
    perm_ensemble_fitnesses.sort(reverse=True, key=lambda x: x['fitness'])
    weighted_population.append(perm_ensemble_fitnesses)


def ranking_correlations(weighted_population, stats):
    old_ensemble_iterations = globals.get('ensemble_iterations')
    fitness_funcs = {'ensemble_fitness': ensemble_fitness, 'normal_fitness': normal_fitness}
    for num_iterations in globals.get('ranking_correlation_num_iterations'):
        rankings = []
        globals.set('ensemble_iterations', num_iterations)
        for fitness_func in globals.get('ranking_correlation_fitness_funcs'):
            weighted_pop_copy = deepcopy(weighted_population)
            for i, pop in enumerate(weighted_pop_copy):
                pop['order'] = i
            fitness_funcs[fitness_func](weighted_pop_copy)
            weighted_pop_copy = sorted(weighted_pop_copy, key=lambda x: x['fitness'], reverse=True)
            ranking = [pop['order'] for pop in weighted_pop_copy]
            rankings.append(ranking)
        correlation = spearmanr(*rankings)
        stats[f'ranking_correlation_{num_iterations}'] = correlation[0]
    globals.set('ensemble_iterations', old_ensemble_iterations)


def sort_population(weighted_population):
    new_weighted_pop = []
    if globals.get('perm_ensembles'):
        ensemble_order = weighted_population[globals.get('pop_size')]
        del weighted_population[globals.get('pop_size')]
        for order in ensemble_order:
            pops = [weighted_population[i] for i in range(globals.get('pop_size'))
                    if weighted_population[i]['group_id'] == order['group_id']]
            new_weighted_pop.extend(pops)
        return new_weighted_pop
    else:
        return sorted(weighted_population, key=lambda x: x['fitness'], reverse=True)


def add_model_to_stats(pop, model_index, model_stats):
    if globals.get('grid'):
        if globals.get('grid_as_ensemble'):
            for key, value in pop['weighted_avg_params'].items():
                model_stats[key] = value
    else:
        for i, layer in enumerate(pop['model']):
            model_stats[f'layer_{i}'] = type(layer).__name__
            for key, value in vars(layer).items():
                model_stats[f'layer_{i}_{key}'] = value
    if globals.get('perm_ensembles'):
        model_stats['ensemble_role'] = (model_index % globals.get('ensemble_size'))
        assert pop['perm_ensemble_role'] == model_stats['ensemble_role']
        model_stats['perm_ensemble_id'] = pop['perm_ensemble_id']
    if globals.get('delete_finalized_models'):
        finalized_model = models_generation.finalize_model(pop['model'])
    else:
        finalized_model = pop['finalized_model']
    model_stats['trainable_params'] = pytorch_count_params(finalized_model)


def train_time_penalty(weighted_population):
    train_time_indices = [i[0] for i in sorted(enumerate
                                               (weighted_population), key=lambda x: x[1]['train_time'])]
    for rank, idx in enumerate(train_time_indices):
        weighted_population[idx]['fitness'] -= (rank / globals.get('pop_size')) *\
                                                weighted_population[idx]['fitness'] * globals.get('penalty_factor')


def pytorch_count_params(model):
    total_params = sum(reduce(lambda a, b: a*b, x.size()) for x in model.parameters())
    return total_params


def format_manual_ensemble_evaluations(avg_evaluations):
    for eval in avg_evaluations.items():
        for eval_spec in eval[1].items():
            if type(eval_spec[1] == list):
                avg_evaluations[eval[0]][eval_spec[0]] = np.mean(eval_spec[1], axis=0)
            else:
                avg_evaluations[eval[0]][eval_spec[0]] = np.mean(eval_spec[1])
    new_avg_evaluations = defaultdict(dict)
    for dataset in ['train', 'valid', 'test']:
        ensemble_preds = avg_evaluations['raw'][dataset]
        pred_labels = np.argmax(ensemble_preds, axis=1).squeeze()
        ensemble_targets = avg_evaluations['target'][dataset]
        ensemble_fit = getattr(utils, f'{globals.get("ga_objective")}_func')(pred_labels, ensemble_targets)
        objective_str = globals.get("ga_objective")
        new_avg_evaluations[f'ensemble_{objective_str}'][dataset] = ensemble_fit
    return new_avg_evaluations


def set_finetuning(model, X):
    child_idx = 0
    num_layers = len(list(model.children()))
    for child in model.children():
        if child_idx < num_layers - X:
            for param in child.parameters():
                param.requires_grad = False
        child_idx += 1


def evaluate_single_model(model, X, y, eval_func):
    if X.ndim == 3:
        X = X[:, :, :, None]
    model.eval()
    with torch.no_grad():
        X = np_to_var(X, pin_memory=globals.get('pin_memory'))
        if torch.cuda.is_available():
            with torch.cuda.device(0):
                X = X.cuda()
        preds = model(X)
        preds = preds.cpu().data.numpy()
        pred_labels = np.argmax(preds, axis=1).squeeze()
        return eval_func(pred_labels, y)




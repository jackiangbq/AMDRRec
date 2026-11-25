
import ast

import torch
import numpy as np
from utility.data_loader import Data
from utility.tools import mini_batch
import utility.metrics


def evaluate_split(dataset: Data, model, device, config, split_name):
    if split_name == "validation":
        ground_truth_dict = dataset.validation_dict
        include_validation_in_exclusions = False
    elif split_name == "test":
        ground_truth_dict = dataset.test_dict
        include_validation_in_exclusions = True
    else:
        raise ValueError("split_name must be validation or test")

    if not ground_truth_dict:
        raise RuntimeError(f"No interactions are available for {split_name} evaluation.")

    model = model.eval()

    topK = tuple(int(k) for k in ast.literal_eval(config.top_K))

    model_results = {'precision': np.zeros(len(topK)),
                     'recall': np.zeros(len(topK)),
                     'hit': np.zeros(len(topK)),
                     'ndcg': np.zeros(len(topK))}

    with torch.no_grad():
        users = list(ground_truth_dict.keys())
        users_list, rating_list, ground_true_list = [], [], []
        batch_size = int(config.test_batch_size)
        num_batch = (len(users) + batch_size - 1) // batch_size


        all_user_embeddings, all_item_embeddings = (
            model.get_deterministic_embeddings_for_test()
        )

        for batch_users in mini_batch(users, batch_size=int(config.test_batch_size)):
            exclude_users, exclude_items = [], []
            all_positive = dataset.get_user_known_items(
                batch_users,
                include_validation=include_validation_in_exclusions,
            )
            ground_true = [ground_truth_dict[u] for u in batch_users]

            batch_users_device = torch.Tensor(batch_users).long().to(device)

            rating = model.get_rating_for_test(
                batch_users_device,
                all_user_embeddings,
                all_item_embeddings
            )


            for i, items in enumerate(all_positive):
                exclude_users.extend([i] * len(items))
                exclude_items.extend(items)
            rating[exclude_users, exclude_items] = -torch.inf


            _, rating_k = torch.topk(rating, k=max(topK))

            rating = rating.cpu()
            del rating

            users_list.append(batch_users)
            rating_list.append(rating_k.cpu())
            ground_true_list.append(ground_true)

        assert num_batch == len(users_list)
        enum_list = zip(rating_list, ground_true_list)

        results = []
        for single_list in enum_list:
            results.append(test_one_batch(single_list, topK))

        for result in results:
            model_results['recall'] += result['recall']
            model_results['precision'] += result['precision']
            model_results['ndcg'] += result['ndcg']

        model_results['recall'] /= float(len(users))
        model_results['precision'] /= float(len(users))
        model_results['ndcg'] /= float(len(users))

        return model_results


def Validation(dataset: Data, model, device, config):
    """Evaluate validation data while excluding training positives only."""
    return evaluate_split(dataset, model, device, config, "validation")


def Test(dataset: Data, model, device, config):
    """Evaluate sealed test data while excluding train and validation positives."""
    return evaluate_split(dataset, model, device, config, "test")


def test_one_batch(X, topK):
    recommender_items = X[0].numpy()
    ground_true_items = X[1]
    r = utility.metrics.get_label(ground_true_items, recommender_items)
    precision, recall, ndcg = [], [], []

    for k_size in topK:
        recall.append(utility.metrics.recall_at_k(r, k_size, ground_true_items))
        precision.append(utility.metrics.precision_at_k(r, k_size, ground_true_items))
        ndcg.append(utility.metrics.ndcg_at_k(r, k_size, ground_true_items))

    return {'recall': np.array(recall), 'precision': np.array(precision), 'ndcg': np.array(ndcg)}

   
def sparsity_test(dataset: Data, model, device, config):
    sparsity_results = []
    model = model.eval()
    # top-20, 40, ..., 100
    topK = tuple(int(k) for k in ast.literal_eval(config.top_K))

    with torch.no_grad():
        # 所有稀疏度分组共享同一份确定性融合表示。
        all_user_embeddings, all_item_embeddings = (
            model.get_deterministic_embeddings_for_test()
        )

        for users in dataset.split_test_dict:
            model_results = {
                'precision': np.zeros(len(topK)),
                'recall': np.zeros(len(topK)),
                'hit': np.zeros(len(topK)),
                'ndcg': np.zeros(len(topK))
            }
            users_list, rating_list, ground_true_list = [], [], []
            batch_size = int(config.test_batch_size)
            num_batch = (len(users) + batch_size - 1) // batch_size

            for batch_users in mini_batch(users, batch_size=int(config.test_batch_size)):
                exclude_users, exclude_items = [], []
                all_positive = dataset.get_user_known_items(
                    batch_users,
                    include_validation=True,
                )
                ground_true = [dataset.test_dict[u] for u in batch_users]

                batch_users_device = torch.Tensor(batch_users).long().to(device)

                rating = model.get_rating_for_test(
                    batch_users_device,
                    all_user_embeddings,
                    all_item_embeddings
                )


                for i, items in enumerate(all_positive):
                    exclude_users.extend([i] * len(items))
                    exclude_items.extend(items)

                rating[exclude_users, exclude_items] = -torch.inf


                _, rating_k = torch.topk(rating, k=max(topK))

                rating = rating.cpu()
                del rating

                users_list.append(batch_users)
                rating_list.append(rating_k.cpu())
                ground_true_list.append(ground_true)

            assert num_batch == len(users_list)
            enum_list = zip(rating_list, ground_true_list)

            results = []
            for single_list in enum_list:
                results.append(test_one_batch(single_list, topK))

            for result in results:
                model_results['recall'] += result['recall']
                model_results['precision'] += result['precision']
                model_results['ndcg'] += result['ndcg']

            model_results['recall'] /= float(len(users))
            model_results['precision'] /= float(len(users))
            model_results['ndcg'] /= float(len(users))
            sparsity_results.append(model_results)

    return sparsity_results

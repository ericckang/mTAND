# pylint: disable=E1101, E0401, E1102, W0621, W0221

import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time

from random import SystemRandom
import models
import utils
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--std', type=float, default=0.01)
parser.add_argument('--latent-dim', type=int, default=32)
parser.add_argument('--rec-hidden', type=int, default=32)
parser.add_argument('--gen-hidden', type=int, default=50)
parser.add_argument('--embed-time', type=int, default=128)
parser.add_argument('--save', type=int, default=1)
parser.add_argument('--enc', type=str, default='mtan_rnn')
parser.add_argument('--dec', type=str, default='mtan_rnn')
parser.add_argument('--fname', type=str, default=None)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--split', type=int, default=0)
parser.add_argument('--n', type=int, default=8000)
parser.add_argument('--batch-size', type=int, default=50)
parser.add_argument('--quantization', type=float, default=0.1,
                    help="Quantization on the physionet dataset.")
parser.add_argument('--classif', action='store_true',
                    help="Include binary classification loss")
parser.add_argument('--freq', type=float, default=10.)
parser.add_argument('--k-iwae', type=int, default=10)
parser.add_argument('--norm', action='store_true')
parser.add_argument('--kl', action='store_true')
parser.add_argument('--learn-emb', action='store_true')
parser.add_argument('--dataset', type=str, default='physionet')
parser.add_argument('--alpha', type=int, default=100.)
parser.add_argument('--old-split', type=int, default=1)
parser.add_argument('--nonormalize', action='store_true')
parser.add_argument('--enc-num-heads', type=int, default=1)
parser.add_argument('--dec-num-heads', type=int, default=1)
parser.add_argument('--num-ref-points', type=int, default=128)
parser.add_argument('--classify-pertp', action='store_true')
args = parser.parse_args()


def extract_representations(model, data_loader, args, device):
    """
    Extracts representations from the encoder model for the given data loader.
    """
    model.eval()
    representations_list = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if isinstance(batch, list) or isinstance(batch, tuple):
                data = batch[0]
            else:
                data = batch
            data = data.to(device)
            observed_data, observed_mask, observed_tp = (
                data[:, :, :args.input_dim],
                data[:, :, args.input_dim:2 * args.input_dim],
                data[:, :, -1],
            )
            # Optionally, print observed_data and observed_mask samples
            if batch_idx == 0:
                print("Observed Data Sample (batch 0):", observed_data[0])
                print("Observed Mask Sample (batch 0):", observed_mask[0])
            _, _, representations = model(torch.cat((observed_data, observed_mask), 2), observed_tp)
            representations_list.append(representations.cpu())
    representations = torch.cat(representations_list, dim=0)
    return representations



def compute_similarity(rep_complete, rep_missing):
    rep_complete_avg = torch.mean(rep_complete, dim=1)
    rep_missing_avg = torch.mean(rep_missing, dim=1)
    cos_sim = torch.nn.functional.cosine_similarity(rep_complete_avg, rep_missing_avg, dim=1)
    mse = torch.mean((rep_complete_avg - rep_missing_avg) ** 2, dim=1)
    euclidean_dist = torch.norm(rep_complete_avg - rep_missing_avg, dim=1)
    return cos_sim.cpu().numpy(), mse.cpu().numpy(), euclidean_dist.cpu().numpy()


if __name__ == '__main__':
    experiment_id = int(SystemRandom().random() * 100000)
    print(args, experiment_id)
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')

    if args.dataset == 'physionet':
        data_obj = utils.get_physionet_data(args, device, args.quantization)
    elif args.dataset == 'mimiciii':
        data_obj = utils.get_mimiciii_data(args)

    train_loader = data_obj["train_dataloader"]
    test_loader = data_obj["test_dataloader"]
    val_loader = data_obj.get("val_dataloader")
    train_data = data_obj["train_data"]
    test_data = data_obj["test_data"]
    dim = data_obj["input_dim"]
    args.input_dim = dim

    if args.enc == 'enc_rnn3':
        rec = models.enc_rnn3(
            dim, torch.linspace(0, 1., args.num_ref_points), args.latent_dim, args.rec_hidden,
            embed_time=args.embed_time, learn_emb=args.learn_emb).to(device)
    elif args.enc == 'mtan_rnn':
        rec = models.enc_mtan_rnn(
            dim, torch.linspace(0, 1., args.num_ref_points), args.latent_dim, args.rec_hidden,
            embed_time=args.embed_time, learn_emb=args.learn_emb, num_heads=args.enc_num_heads).to(device)

    classifier = models.create_classifier(args.latent_dim, args.rec_hidden).to(device)
    params = list(rec.parameters()) + list(classifier.parameters())
    optimizer = optim.Adam(params, lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    if args.fname is not None:
        checkpoint = torch.load(args.fname)
        rec.load_state_dict(checkpoint['rec_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('loading saved weights', checkpoint['epoch'])

    best_val_loss = float('inf')
    total_time = 0.
    for itr in range(1, args.niters + 1):
        rec.train()
        train_ce_loss = 0
        train_n = 0
        train_acc = 0
        start_time = time.time()
        for train_batch, label in train_loader:
            train_batch, label = train_batch.to(device), label.to(device)
            batch_len = train_batch.shape[0]
            observed_data, observed_mask, observed_tp = (
                train_batch[:, :, :dim],
                train_batch[:, :, dim:2 * dim],
                train_batch[:, :, -1],
            )
            qz0_mean, qz0_logvar, _ = rec(torch.cat((observed_data, observed_mask), 2), observed_tp)
            # Sample z0
            epsilon = torch.randn_like(qz0_mean).to(device)
            z0 = epsilon * torch.exp(0.5 * qz0_logvar) + qz0_mean
            # Use representation from the last time step
            pred_y = classifier(z0[:, -1:, :]) 
            # Compute loss
            print("pred_y shape:", pred_y.shape)
            print("label shape:", label.shape)
            ce_loss = criterion(pred_y, label)
            loss = ce_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_ce_loss += ce_loss.item() * batch_len
            train_acc += (pred_y.argmax(1) == label).sum().item()
            train_n += batch_len
        total_time += time.time() - start_time
        # Validation step
        if val_loader is not None:
            val_loss, val_acc, _ = utils.evaluate_classifier(
                rec, val_loader, args=args, classifier=classifier, num_sample=1, dim=dim)
            if val_loss <= best_val_loss:
                best_val_loss = val_loss
                best_state = {
                    'rec_state_dict': rec.state_dict(),
                    'classifier_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
            print('Iter: {}, ce_loss: {:.4f}, acc: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}'.format(
                itr, train_ce_loss / train_n, train_acc / train_n, val_loss, val_acc))
        else:
            print('Iter: {}, ce_loss: {:.4f}, acc: {:.4f}'.format(
                itr, train_ce_loss / train_n, train_acc / train_n))

    # Load the best model
    rec.load_state_dict(best_state['rec_state_dict'])
    classifier.load_state_dict(best_state['classifier_state_dict'])

    # After loading data_obj and train_data
    train_data = data_obj["train_data"]
    #print(train_data)
    # Deep copy
    train_data_original = copy.deepcopy(train_data)
    #print(train_data_original)
    # Compute data_min and data_max from train_data
    data_min, data_max = utils.get_data_min_max(train_data_original, device)

    # Preprocess the complete data
    complete_train_data_combined, complete_labels = utils.variable_time_collate_fn(
        train_data_original, device, classify=args.classif, data_min=data_min, data_max=data_max)

    # Create TensorDataset and DataLoader for complete data
    complete_train_dataset = TensorDataset(complete_train_data_combined, complete_labels.long().squeeze())
    complete_train_dataloader = DataLoader(
        complete_train_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    # Extract representations from the complete dataset
    print("Extracting representations from complete data...")
    rec.eval()
    classifier.eval()
    rep_complete = extract_representations(rec, complete_train_dataloader, args, device)

    # Introduce missingness and compute similarities
    missingness_scenarios = {
    'Random Missingness': utils.introduce_missingness_random,
    'Chunk Missingness': utils.introduce_missingness_chunks,
    'Mixed Missingness': utils.introduce_missingness_mixed
    }

    missing_rates = [0.1, 0.2, 0.3, 0.4, 0.5]
    results = {}
    output_filename = 'results.txt'
    with open(output_filename, 'w') as f:
        for scenario_name, missingness_function in missingness_scenarios.items():
            print(f"Processing scenario: {scenario_name}")
            f.write(f"Processing scenario: {scenario_name}\n")
            results[scenario_name] = []
            for rate in missing_rates:
                print(f"  Missingness rate: {rate}")
                f.write(f"  Missingness rate: {rate}\n")
                # Introduce missingness using the appropriate function
                missing_train_data = missingness_function(copy.deepcopy(train_data_original), rate)
                # Create dataloader for missing data
                missing_train_data_combined, missing_labels = utils.variable_time_collate_fn(
                    missing_train_data, device, classify=args.classif, data_min=data_min, data_max=data_max)
                missing_train_dataloader = DataLoader(
                    TensorDataset(missing_train_data_combined, missing_labels.long().squeeze()),
                    batch_size=args.batch_size, shuffle=False)
                # Extract representations from missing data
                rep_missing = extract_representations(rec, missing_train_dataloader, args, device)
                # Compute similarities
                cos_sim, mse, euclidean_dist = compute_similarity(rep_complete, rep_missing)
                # Store results
                results[scenario_name].append({
                    'Missing Rate': rate,
                    'Cosine Similarity Mean': np.mean(cos_sim),
                    'Cosine Similarity Std': np.std(cos_sim),
                    'MSE Mean': np.mean(mse),
                    'MSE Std': np.std(mse),
                    'Euclidean Distance Mean': np.mean(euclidean_dist),
                    'Euclidean Distance Std': np.std(euclidean_dist),
                    'Scenario': scenario_name
                })
                print(f"    Cosine Similarity: Mean = {np.mean(cos_sim):.8f}, Std = {np.std(cos_sim):.8f}")
                print(f"    MSE: Mean = {np.mean(mse):.8f}, Std = {np.std(mse):.8f}")
                print(f"    Euclidean Distance: Mean = {np.mean(euclidean_dist):.8f}, Std = {np.std(euclidean_dist):.8f}")
                f.write(f"    Cosine Similarity: Mean = {np.mean(cos_sim):.8f}, Std = {np.std(cos_sim):.8f}")
                f.write(f"    MSE: Mean = {np.mean(mse):.8f}, Std = {np.std(mse):.8f}")
                f.write(f"    Euclidean Distance: Mean = {np.mean(euclidean_dist):.8f}, Std = {np.std(euclidean_dist):.8f}")
        all_results = []
        for scenario_name in results:
            scenario_results = results[scenario_name]
            all_results.extend(scenario_results)
        # Create a DataFrame from all results
        df_all = pd.DataFrame(all_results)

        # Write the combined results to the text file
        f.write("\nCombined Results:\n")
        f.write(df_all.to_string(index=False))
        f.write('\n')

    colors = {'Random Missingness': 'blue', 'Chunk Missingness': 'green', 'Mixed Missingness': 'red'}
    markers = {'Random Missingness': 'o', 'Chunk Missingness': 's', 'Mixed Missingness': '^'}
    linestyles = {'Random Missingness': '-', 'Chunk Missingness': '--', 'Mixed Missingness': '-.'}

    metrics = ['Cosine Similarity Mean', 'MSE Mean', 'Euclidean Distance Mean']
    y_labels = ['Cosine Similarity', 'MSE', 'Euclidean Distance']

    for metric, y_label in zip(metrics, y_labels):
        plt.figure(figsize=(10, 6))
        for scenario_name in results:
            df_scenario = df_all[df_all['Scenario'] == scenario_name]
            plt.plot(df_scenario['Missing Rate'], df_scenario[metric],
                     label=scenario_name,
                     color=colors[scenario_name],
                     marker=markers[scenario_name],
                     linestyle=linestyles[scenario_name])
        plt.title(f'{y_label} vs Missing Rate for Different Missingness Models')
        plt.xlabel('Missing Rate')
        plt.ylabel(y_label)
        plt.legend(title='Missingness Scenario')
        plt.grid(True)
        # Adjust y-axis limits if necessary
        if metric == 'Cosine Similarity Mean':
            plt.ylim(0.995, 1.0)
        elif metric == 'MSE Mean':
            plt.ylim(0, df_all[metric].max() * 1.1)
        elif metric == 'Euclidean Distance Mean':
            plt.ylim(0, df_all[metric].max() * 1.1)
        plt.tight_layout()
        # Save the figure as an image file
        plt.savefig(f'{metric.replace(" ", "_")}_comparison.png')
        plt.show()
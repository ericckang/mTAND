# pylint: disable=E1101
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import random

import numpy as np
from physionet import PhysioNet, get_data_min_max, variable_time_collate_fn2
from sklearn import model_selection
from sklearn import metrics
from person_activity import PersonActivity

def fill_missing_values_with_mean(dataset, device):
    """
    Fills missing values in the dataset with the mean value for each variable.
    """
    # dataset is a list of (record_id, tt, vals, mask, labels)
    input_dim = dataset[0][2].shape[1]
    sum_vals = torch.zeros(input_dim).to(device)
    count_vals = torch.zeros(input_dim).to(device)
    
    # Calculate mean values for each feature across the dataset
    for (_, _, vals, mask, _) in dataset:
        sum_vals += torch.sum(vals * mask, dim=0)
        count_vals += torch.sum(mask, dim=0)
    mean_vals = sum_vals / count_vals
    mean_vals[torch.isnan(mean_vals)] = 0.0  # Handle NaNs

    # Fill missing values with mean
    filled_dataset = []
    for (record_id, tt, vals, mask, labels) in dataset:
        vals = vals.clone()
        mask = mask.clone()
        missing_mask = (mask == 0)
        # Expand mean_vals to match the shape of vals
        mean_vals_expanded = mean_vals.unsqueeze(0).expand_as(vals)
        # Assign mean values to missing positions
        vals[missing_mask] = mean_vals_expanded[missing_mask]
        # Update mask to indicate that missing values have been filled
        mask[missing_mask] = 1
        filled_dataset.append((record_id, tt, vals, mask, labels))
    return filled_dataset

def introduce_missingness_random(dataset, missing_rate):
    new_dataset = []
    for record_id, tt, vals, mask, labels in dataset:
        vals = vals.clone()
        mask = mask.clone()
        total_values = mask.sum().item()
        num_missing = int(total_values * missing_rate)
        present_indices = torch.nonzero(mask, as_tuple=False)
        missing_indices = present_indices[torch.randperm(len(present_indices))[:num_missing]]
        vals[missing_indices[:, 0], missing_indices[:, 1]] = 0.0
        mask[missing_indices[:, 0], missing_indices[:, 1]] = 0
        new_dataset.append((record_id, tt, vals, mask, labels))
    return new_dataset

def introduce_missingness_chunks(dataset, missing_rate):
    new_dataset = []
    total_values_being_zeroed = []
    total_indices_being_zeroed = []
    for idx, (record_id, tt, vals, mask, labels) in enumerate(dataset):
        vals = vals.clone()
        mask = mask.clone()
        total_time_steps = len(tt)
        num_variables = vals.shape[1]
        # Number of observed values
        total_values = int(mask.sum().item())
        # Calculate the number of time steps to remove
        num_missing_time_steps = int(total_time_steps * missing_rate)
        if num_missing_time_steps < 1:
            num_missing_time_steps = 1
        if num_missing_time_steps >= total_time_steps:
            num_missing_time_steps = total_time_steps - 1

        # Randomly select the start index for the missing chunk
        start_idx = random.randint(0, total_time_steps - num_missing_time_steps)
        end_idx = start_idx + num_missing_time_steps

        # Set mask and values to zero for the missing time steps
        vals[start_idx:end_idx, :] = 0.0
        mask[start_idx:end_idx, :] = 0

        # Store information about the missing values
        missing_vals = vals[start_idx:end_idx, :][mask[start_idx:end_idx, :] == 0]
        total_values_being_zeroed.extend(missing_vals.cpu().numpy())
        indices_being_zeroed = torch.nonzero(mask[start_idx:end_idx, :] == 0, as_tuple=False)
        indices_being_zeroed[:, 0] += start_idx  # Adjust indices
        total_indices_being_zeroed.extend(indices_being_zeroed.cpu().numpy())

        new_dataset.append((record_id, tt, vals, mask, labels))

    # After processing all records, print the information
    print(f"Introduced missingness: {missing_rate*100}% missing time steps.")
    total_values_being_zeroed = np.array(total_values_being_zeroed)
    total_indices_being_zeroed = np.array(total_indices_being_zeroed)
    print(f"Number of values being set to zero: {len(total_values_being_zeroed)}")
    print("Sample of values being set to zero:")
    print(total_values_being_zeroed[:10])
    print("Corresponding indices:")
    print(total_indices_being_zeroed[:10])

    return new_dataset

def introduce_missingness_mixed(dataset, missing_rate):
    new_dataset = []
    for record_id, tt, vals, mask, labels in dataset:
        vals = vals.clone()
        mask = mask.clone()
        # Apply chunk missingness to half of the missing_rate
        chunk_missing_rate = missing_rate / 2
        total_time_steps = len(tt)
        num_missing_time_steps = int(total_time_steps * chunk_missing_rate)
        num_missing_time_steps = max(1, min(num_missing_time_steps, total_time_steps - 1))
        start_idx = random.randint(0, total_time_steps - num_missing_time_steps)
        end_idx = start_idx + num_missing_time_steps
        vals[start_idx:end_idx, :] = 0.0
        mask[start_idx:end_idx, :] = 0
        # Apply random missingness to the remaining half
        random_missing_rate = missing_rate / 2
        total_values = mask.sum().item()
        num_missing = int(total_values * random_missing_rate)
        present_indices = torch.nonzero(mask, as_tuple=False)
        if len(present_indices) > num_missing:
            missing_indices = present_indices[torch.randperm(len(present_indices))[:num_missing]]
            vals[missing_indices[:, 0], missing_indices[:, 1]] = 0.0
            mask[missing_indices[:, 0], missing_indices[:, 1]] = 0
        new_dataset.append((record_id, tt, vals, mask, labels))
    return new_dataset


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def log_normal_pdf(x, mean, logvar, mask):
    const = torch.from_numpy(np.array([2. * np.pi])).float().to(x.device)
    const = torch.log(const)
    return -.5 * (const + logvar + (x - mean) ** 2. / torch.exp(logvar)) * mask


def normal_kl(mu1, lv1, mu2, lv2):
    v1 = torch.exp(lv1)
    v2 = torch.exp(lv2)
    lstd1 = lv1 / 2.
    lstd2 = lv2 / 2.

    kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.) / (2. * v2)) - .5
    return kl


def mean_squared_error(orig, pred, mask):
    error = (orig - pred) ** 2
    error = error * mask
    return error.sum() / mask.sum()


def normalize_masked_data(data, mask, att_min, att_max):
    att_max[att_max == 0.] = 1.
    data_norm = (data - att_min) / att_max
    # Do not replace NaNs with zeros
    # data_norm = torch.nan_to_num(data_norm, nan=0.0)
    # Instead, keep NaNs or replace them with a distinct value
    data_norm[mask == 0] = float('nan')  # or data_norm[mask == 0] = -1e9
    return data_norm, att_min, att_max



def evaluate(dim, rec, dec, test_loader, args, num_sample=10, device="cuda"):
    mse, test_n = 0.0, 0.0
    with torch.no_grad():
        for test_batch in test_loader:
            test_batch = test_batch.to(device)
            observed_data, observed_mask, observed_tp = (
                test_batch[:, :, :dim],
                test_batch[:, :, dim: 2 * dim],
                test_batch[:, :, -1],
            )
            if args.sample_tp and args.sample_tp < 1:
                subsampled_data, subsampled_tp, subsampled_mask = subsample_timepoints(
                    observed_data.clone(), observed_tp.clone(), observed_mask.clone(), args.sample_tp)
            else:
                subsampled_data, subsampled_tp, subsampled_mask = \
                    observed_data, observed_tp, observed_mask
            out = rec(torch.cat((subsampled_data, subsampled_mask), 2), subsampled_tp)
            qz0_mean, qz0_logvar = (
                out[:, :, : args.latent_dim],
                out[:, :, args.latent_dim:],
            )
            epsilon = torch.randn(
                num_sample, qz0_mean.shape[0], qz0_mean.shape[1], qz0_mean.shape[2]
            ).to(device)
            z0 = epsilon * torch.exp(0.5 * qz0_logvar) + qz0_mean
            z0 = z0.view(-1, qz0_mean.shape[1], qz0_mean.shape[2])
            batch, seqlen = observed_tp.size()
            time_steps = (
                observed_tp[None, :, :].repeat(num_sample, 1, 1).view(-1, seqlen)
            )
            pred_x = dec(z0, time_steps)
            pred_x = pred_x.view(num_sample, -1, pred_x.shape[1], pred_x.shape[2])
            pred_x = pred_x.mean(0)
            mse += mean_squared_error(observed_data, pred_x, observed_mask) * batch
            test_n += batch
    return mse / test_n


def compute_losses(dim, dec_train_batch, qz0_mean, qz0_logvar, pred_x, args, device):
    observed_data, observed_mask \
        = dec_train_batch[:, :, :dim], dec_train_batch[:, :, dim:2*dim]

    noise_std = args.std  # default 0.1
    noise_std_ = torch.zeros(pred_x.size()).to(device) + noise_std
    noise_logvar = 2. * torch.log(noise_std_).to(device)
    logpx = log_normal_pdf(observed_data, pred_x, noise_logvar,
                           observed_mask).sum(-1).sum(-1)
    pz0_mean = pz0_logvar = torch.zeros(qz0_mean.size()).to(device)
    analytic_kl = normal_kl(qz0_mean, qz0_logvar,
                            pz0_mean, pz0_logvar).sum(-1).sum(-1)
    if args.norm:
        logpx /= observed_mask.sum(-1).sum(-1)
        analytic_kl /= observed_mask.sum(-1).sum(-1)
    return logpx, analytic_kl


def evaluate_classifier(rec, data_loader, args, classifier, num_sample=1, dim=1):
    rec.eval()
    classifier.eval()
    test_loss = 0
    test_acc = 0
    test_n = 0
    with torch.no_grad():
        for batch in data_loader:
            data, label = batch[0].to(rec.device), batch[1].to(rec.device)
            batch_len = data.size(0)
            observed_data = data[:, :, :dim]
            observed_mask = data[:, :, dim:2*dim]
            observed_tp = data[:, :, -1]
            qz0_mean, qz0_logvar, _ = rec(torch.cat((observed_data, observed_mask), 2), observed_tp)
            z0 = qz0_mean
            pred_y = classifier(z0[:, -1:, :])  # Ensure the input has the correct shape
            # **If pred_y is a tuple, extract the first element**
            if isinstance(pred_y, tuple):
                pred_y = pred_y[0]
            # **Check the type and shape of pred_y**
            print(f"Type of pred_y: {type(pred_y)}")
            print(f"Shape of pred_y: {pred_y.shape}")
            test_loss += nn.CrossEntropyLoss()(pred_y, label).item() * batch_len * num_sample
            test_acc += (pred_y.argmax(1) == label).sum().item() * num_sample
            test_n += batch_len * num_sample
    test_loss /= test_n
    test_acc /= test_n
    return test_loss, test_acc, test_n



def get_mimiciii_data(args):
    input_dim = 12
    x = np.load('../../../neuraltimeseries/Dataset/final_input3.npy')
    y = np.load('../../../neuraltimeseries/Dataset/final_output3.npy')
    x = x[:, :25]
    x = np.transpose(x, (0, 2, 1))

    # normalize values and time
    observed_vals, observed_mask, observed_tp = x[:, :,
                                                  :input_dim], x[:, :, input_dim:2*input_dim], x[:, :, -1]
    if np.max(observed_tp) != 0.:
        observed_tp = observed_tp / np.max(observed_tp)

    if not args.nonormalize:
        for k in range(input_dim):
            data_min, data_max = float('inf'), 0.
            for i in range(observed_vals.shape[0]):
                for j in range(observed_vals.shape[1]):
                    if observed_mask[i, j, k]:
                        data_min = min(data_min, observed_vals[i, j, k])
                        data_max = max(data_max, observed_vals[i, j, k])
            #print(data_min, data_max)
            if data_max == 0:
                data_max = 1
            observed_vals[:, :, k] = (
                observed_vals[:, :, k] - data_min)/data_max
    # set masked out elements back to zero
    observed_vals[observed_mask == 0] = 0
    print(observed_vals[0], observed_tp[0])
    print(x.shape, y.shape)
    kfold = model_selection.StratifiedKFold(
        n_splits=5, shuffle=True, random_state=0)
    splits = [(train_inds, test_inds)
              for train_inds, test_inds in kfold.split(np.zeros(len(y)), y)]
    x_train, y_train = x[splits[args.split][0]], y[splits[args.split][0]]
    test_data_x, test_data_y = x[splits[args.split]
                                 [1]], y[splits[args.split][1]]
    if not args.old_split:
        train_data_x, val_data_x, train_data_y, val_data_y = \
            model_selection.train_test_split(
                x_train, y_train, stratify=y_train, test_size=0.2, random_state=0)
    else:
        frac = int(0.8*x_train.shape[0])
        train_data_x, val_data_x = x_train[:frac], x_train[frac:]
        train_data_y, val_data_y = y_train[:frac], y_train[frac:]

    print(train_data_x.shape, train_data_y.shape, val_data_x.shape, val_data_y.shape,
          test_data_x.shape, test_data_y.shape)
    print(np.sum(test_data_y))
    train_data_combined = TensorDataset(torch.from_numpy(train_data_x).float(),
                                        torch.from_numpy(train_data_y).long().squeeze())
    val_data_combined = TensorDataset(torch.from_numpy(val_data_x).float(),
                                      torch.from_numpy(val_data_y).long().squeeze())
    test_data_combined = TensorDataset(torch.from_numpy(test_data_x).float(),
                                       torch.from_numpy(test_data_y).long().squeeze())
    train_dataloader = DataLoader(
        train_data_combined, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(
        test_data_combined, batch_size=args.batch_size, shuffle=False)
    val_dataloader = DataLoader(
        val_data_combined, batch_size=args.batch_size, shuffle=False)

    data_objects = {"train_dataloader": train_dataloader,
                    "test_dataloader": test_dataloader,
                    "val_dataloader": val_dataloader,
                    "input_dim": input_dim}
    return data_objects


def get_physionet_data(args, device, q, flag=1):
    # Load only Set A (training data)
    train_dataset_obj = PhysioNet('data/physionet', train=True,
                                  quantization=q,
                                  download=True, n_samples=min(10000, args.n),
                                  device=device)

    # Use only patients with outcome labels from Outcomes-a.txt
    total_dataset = []
    for data_point in train_dataset_obj:
        record_id, tt, vals, mask, labels = data_point
        if labels is not None:
            total_dataset.append(data_point)

    print(f"Total dataset size with labels: {len(total_dataset)}")

    # Proceed to split the dataset
    train_data, test_data = model_selection.train_test_split(
        total_dataset, train_size=0.8, random_state=42, shuffle=True)
    train_data, val_data = model_selection.train_test_split(
        train_data, train_size=0.8, random_state=11, shuffle=True)

    # Prepare data loaders
    batch_size = min(len(train_data), args.batch_size)
    input_dim = total_dataset[0][2].shape[1]

    # Compute data_min and data_max from the dataset
    data_min, data_max = get_data_min_max(total_dataset, device)

    # Pass data_min and data_max to variable_time_collate_fn
    train_data_combined, train_labels = variable_time_collate_fn(
        train_data, device, classify=args.classif, data_min=data_min, data_max=data_max)
    val_data_combined, val_labels = variable_time_collate_fn(
        val_data, device, classify=args.classif, data_min=data_min, data_max=data_max)
    test_data_combined, test_labels = variable_time_collate_fn(
        test_data, device, classify=args.classif, data_min=data_min, data_max=data_max)

    train_data_combined = TensorDataset(
        train_data_combined, train_labels.long().squeeze())
    val_data_combined = TensorDataset(
        val_data_combined, val_labels.long().squeeze())
    test_data_combined = TensorDataset(
        test_data_combined, test_labels.long().squeeze())

    train_dataloader = DataLoader(
        train_data_combined, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(
        val_data_combined, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(
        test_data_combined, batch_size=batch_size, shuffle=False)

    data_objects = {
        "dataset_obj": total_dataset,
        "train_data": train_data,
        "val_data": val_data,
        "test_data": test_data,
        "train_dataloader": train_dataloader,
        "val_dataloader": val_dataloader,
        "test_dataloader": test_dataloader,
        "input_dim": input_dim,
    }

    return data_objects

def variable_time_collate_fn(batch, device=torch.device("cpu"), classify=False, activity=False,
                             data_min=None, data_max=None):
    """
    Expects a batch of time series data in the form of (record_id, tt, vals, mask, labels) where
      - record_id is a patient id
      - tt is a 1-dimensional tensor containing T time values of observations.
      - vals is a (T, D) tensor containing observed values for D variables.
      - mask is a (T, D) tensor containing 1 where values were observed and 0 otherwise.
      - labels is a list of labels for the current patient, if labels are available. Otherwise None.
    Returns:
      combined_data: (batch_size, max_seq_len, input_dim * 2 + 1) tensor containing the observed values, masks, and time steps.
      combined_labels: Labels corresponding to each sample in the batch.
    """
    D = batch[0][2].shape[1]
    # number of labels
    N = batch[0][-1].shape[1] if activity else 1
    len_tt = [ex[1].size(0) for ex in batch]
    maxlen = np.max(len_tt)
    enc_combined_tt = torch.zeros([len(batch), maxlen]).to(device)
    enc_combined_vals = torch.zeros([len(batch), maxlen, D]).to(device)
    enc_combined_mask = torch.zeros([len(batch), maxlen, D]).to(device)
    if classify:
        if activity:
            combined_labels = torch.zeros([len(batch), maxlen, N]).to(device)
        else:
            combined_labels = torch.zeros([len(batch), N]).to(device)

    for b, (record_id, tt, vals, mask, labels) in enumerate(batch):
        currlen = tt.size(0)
        enc_combined_tt[b, :currlen] = tt.to(device)
        enc_combined_vals[b, :currlen] = vals.to(device)
        enc_combined_mask[b, :currlen] = mask.to(device)
        if classify:
            if activity:
                combined_labels[b, :currlen] = labels.to(device)
            else:
                combined_labels[b] = labels.to(device)

    if not activity:
        enc_combined_vals, _, _ = normalize_masked_data(enc_combined_vals, enc_combined_mask,
                                                        att_min=data_min, att_max=data_max)
    # Replace NaNs with zeros if necessary
    enc_combined_vals = torch.nan_to_num(enc_combined_vals, nan=0.0)

    if torch.max(enc_combined_tt) != 0.:
        enc_combined_tt = enc_combined_tt / torch.max(enc_combined_tt)

    combined_data = torch.cat(
        (enc_combined_vals, enc_combined_mask, enc_combined_tt.unsqueeze(-1)), 2)
    if classify:
        return combined_data, combined_labels
    else:
        return combined_data


def get_activity_data(args, device):
    n_samples = min(10000, args.n)
    dataset_obj = PersonActivity('data/PersonActivity',
                                 download=True, n_samples=n_samples, device=device)

    print(dataset_obj)

    train_data, test_data = model_selection.train_test_split(dataset_obj, train_size=0.8,
                                                             random_state=42, shuffle=True)

    # train_data = [train_data[i] for i in np.random.choice(len(train_data), len(train_data))]
    # test_data = [test_data[i] for i in np.random.choice(len(test_data), len(test_data))]

    record_id, tt, vals, mask, labels = train_data[0]
    input_dim = vals.size(-1)

    batch_size = min(min(len(dataset_obj), args.batch_size), args.n)
    test_data_combined = variable_time_collate_fn(test_data, device, classify=args.classif,
                                                  activity=True)
    train_data, val_data = model_selection.train_test_split(train_data, train_size=0.8,
                                                            random_state=11, shuffle=True)
    train_data_combined = variable_time_collate_fn(
        train_data, device, classify=args.classif, activity=True)
    val_data_combined = variable_time_collate_fn(
        val_data, device, classify=args.classif, activity=True)
    print(train_data_combined[1].sum(
    ), val_data_combined[1].sum(), test_data_combined[1].sum())
    print(train_data_combined[0].size(), train_data_combined[1].size(),
          val_data_combined[0].size(), val_data_combined[1].size(),
          test_data_combined[0].size(), test_data_combined[1].size())

    train_data_combined = TensorDataset(
        train_data_combined[0], train_data_combined[1].long())
    val_data_combined = TensorDataset(
        val_data_combined[0], val_data_combined[1].long())
    test_data_combined = TensorDataset(
        test_data_combined[0], test_data_combined[1].long())

    train_dataloader = DataLoader(
        train_data_combined, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(
        test_data_combined, batch_size=batch_size, shuffle=False)
    val_dataloader = DataLoader(
        val_data_combined, batch_size=batch_size, shuffle=False)

    #attr_names = train_dataset_obj.params
    data_objects = {"train_dataloader": train_dataloader,
                    "test_dataloader": test_dataloader,
                    "val_dataloader": val_dataloader,
                    "input_dim": input_dim,
                    "n_train_batches": len(train_dataloader),
                    "n_test_batches": len(test_dataloader),
                    # "attr": attr_names, #optional
                    "classif_per_tp": False,  # optional
                    "n_labels": 1}  # optional

    return data_objects


def irregularly_sampled_data_gen(n=10, length=20, seed=0):
    np.random.seed(seed)
    # obs_times = obs_times_gen(n)
    obs_values, ground_truth, obs_times = [], [], []
    for i in range(n):
        t1 = np.sort(np.random.uniform(low=0.0, high=1.0, size=length))
        t2 = np.sort(np.random.uniform(low=0.0, high=1.0, size=length))
        t3 = np.sort(np.random.uniform(low=0.0, high=1.0, size=length))
        a = 10 * np.random.randn()
        b = 10 * np.random.rand()
        f1 = .8 * np.sin(20*(t1+a) + np.sin(20*(t1+a))) + \
            0.01 * np.random.randn()
        f2 = -.5 * np.sin(20*(t2+a + 20) + np.sin(20*(t2+a + 20))
                          ) + 0.01 * np.random.randn()
        f3 = np.sin(12*(t3+b)) + 0.01 * np.random.randn()
        obs_times.append(np.stack((t1, t2, t3), axis=0))
        obs_values.append(np.stack((f1, f2, f3), axis=0))
        #obs_values.append([f1.tolist(), f2.tolist(), f3.tolist()])
        t = np.linspace(0, 1, 100)
        fg1 = .8 * np.sin(20*(t+a) + np.sin(20*(t+a)))
        fg2 = -.5 * np.sin(20*(t+a + 20) + np.sin(20*(t+a + 20)))
        fg3 = np.sin(12*(t+b))
        #ground_truth.append([f1.tolist(), f2.tolist(), f3.tolist()])
        ground_truth.append(np.stack((fg1, fg2, fg3), axis=0))
    return obs_values, ground_truth, obs_times


def sine_wave_data_gen(args, seed=0):
    np.random.seed(seed)
    obs_values, ground_truth, obs_times = [], [], []
    for _ in range(args.n):
        t = np.sort(np.random.choice(np.linspace(
            0, 1., 101), size=args.length, replace=True))
        b = 10 * np.random.rand()
        f = np.sin(12*(t+b)) + 0.1 * np.random.randn()
        obs_times.append(t)
        obs_values.append(f)
        tc = np.linspace(0, 1, 100)
        fg = np.sin(12*(tc + b))
        ground_truth.append(fg)

    obs_values = np.array(obs_values)
    obs_times = np.array(obs_times)
    ground_truth = np.array(ground_truth)
    print(obs_values.shape, obs_times.shape, ground_truth.shape)
    mask = np.ones_like(obs_values)
    combined_data = np.concatenate((np.expand_dims(obs_values, axis=2), np.expand_dims(
        mask, axis=2), np.expand_dims(obs_times, axis=2)), axis=2)
    print(combined_data.shape)
    print(combined_data[0])
    train_data, test_data = model_selection.train_test_split(combined_data, train_size=0.8,
                                                             random_state=42, shuffle=True)
    print(train_data.shape, test_data.shape)
    train_dataloader = DataLoader(torch.from_numpy(
        train_data).float(), batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(torch.from_numpy(
        test_data).float(), batch_size=args.batch_size, shuffle=False)
    data_objects = {"dataset_obj": combined_data,
                    "train_dataloader": train_dataloader,
                    "test_dataloader": test_dataloader,
                    "input_dim": 1,
                    "ground_truth": np.array(ground_truth)}
    return data_objects


def kernel_smoother_data_gen(args, alpha=100., seed=0, ref_points=10):
    np.random.seed(seed)
    obs_values, ground_truth, obs_times = [], [], []
    for _ in range(args.n):
        key_values = np.random.randn(ref_points)
        key_points = np.linspace(0, 1, ref_points)

        query_points = np.sort(np.random.choice(
            np.linspace(0, 1., 101), size=args.length, replace=True))
        # query_points = np.sort(np.random.uniform(low=0.0, high=1.0, size=args.length))
        weights = np.exp(-alpha*(np.expand_dims(query_points,
                                                1) - np.expand_dims(key_points, 0))**2)
        weights /= weights.sum(1, keepdims=True)
        query_values = np.dot(weights, key_values)
        obs_values.append(query_values)
        obs_times.append(query_points)

        query_points = np.linspace(0, 1, 100)
        weights = np.exp(-alpha*(np.expand_dims(query_points,
                                                1) - np.expand_dims(key_points, 0))**2)
        weights /= weights.sum(1, keepdims=True)
        query_values = np.dot(weights, key_values)
        ground_truth.append(query_values)

    obs_values = np.array(obs_values)
    obs_times = np.array(obs_times)
    ground_truth = np.array(ground_truth)
    print(obs_values.shape, obs_times.shape, ground_truth.shape)
    mask = np.ones_like(obs_values)
    combined_data = np.concatenate((np.expand_dims(obs_values, axis=2), np.expand_dims(
        mask, axis=2), np.expand_dims(obs_times, axis=2)), axis=2)
    print(combined_data.shape)
    print(combined_data[0])
    train_data, test_data = model_selection.train_test_split(combined_data, train_size=0.8,
                                                             random_state=42, shuffle=True)
    print(train_data.shape, test_data.shape)
    train_dataloader = DataLoader(torch.from_numpy(
        train_data).float(), batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(torch.from_numpy(
        test_data).float(), batch_size=args.batch_size, shuffle=False)
    data_objects = {"dataset_obj": combined_data,
                    "train_dataloader": train_dataloader,
                    "test_dataloader": test_dataloader,
                    "input_dim": 1,
                    "ground_truth": np.array(ground_truth)}
    return data_objects


def get_toy_data(args):
    dim = 3
    obs_values, ground_truth, obs_times = irregularly_sampled_data_gen(
        args.n, args.length)
    obs_times = np.array(obs_times).reshape(args.n, -1)
    obs_values = np.array(obs_values)
    combined_obs_values = np.zeros((args.n, dim, obs_times.shape[-1]))
    mask = np.zeros((args.n, dim, obs_times.shape[-1]))
    for i in range(dim):
        combined_obs_values[:, i, i *
                            args.length: (i+1)*args.length] = obs_values[:, i]
        mask[:, i, i*args.length: (i+1)*args.length] = 1.
    #print(combined_obs_values.shape, mask.shape, obs_times.shape, np.expand_dims(obs_times, axis=1).shape)
    combined_data = np.concatenate(
        (combined_obs_values, mask, np.expand_dims(obs_times, axis=1)), axis=1)
    combined_data = np.transpose(combined_data, (0, 2, 1))
    print(combined_data.shape)
    train_data, test_data = model_selection.train_test_split(combined_data, train_size=0.8,
                                                             random_state=42, shuffle=True)
    print(train_data.shape, test_data.shape)
    train_dataloader = DataLoader(torch.from_numpy(
        train_data).float(), batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(torch.from_numpy(
        test_data).float(), batch_size=args.batch_size, shuffle=False)
    data_objects = {"dataset_obj": combined_data,
                    "train_dataloader": train_dataloader,
                    "test_dataloader": test_dataloader,
                    "input_dim": dim,
                    "ground_truth": np.array(ground_truth)}
    return data_objects


def compute_pertp_loss(label_predictions, true_label, mask):
    criterion = nn.CrossEntropyLoss(reduction='none')
    n_traj, n_tp, n_dims = label_predictions.size()
    label_predictions = label_predictions.reshape(n_traj * n_tp, n_dims)
    true_label = true_label.reshape(n_traj * n_tp, n_dims)
    mask = torch.sum(mask, -1) > 0
    mask = mask.reshape(n_traj * n_tp,  1)
    _, true_label = true_label.max(-1)
    ce_loss = criterion(label_predictions, true_label.long())
    ce_loss = ce_loss * mask
    return torch.sum(ce_loss)/mask.sum()


def get_physionet_data_extrap(args, device, q, flag=1):
    train_dataset_obj = PhysioNet('data/physionet', train=True,
                                  quantization=q,
                                  download=True, n_samples=min(10000, args.n),
                                  device=device)
    # Use custom collate_fn to combine samples with arbitrary time observations.
    # Returns the dataset along with mask and time steps
    test_dataset_obj = PhysioNet('data/physionet', train=False,
                                 quantization=q,
                                 download=True, n_samples=min(10000, args.n),
                                 device=device)

    # Combine and shuffle samples from physionet Train and physionet Test
    total_dataset = train_dataset_obj[:len(train_dataset_obj)]

    if not args.classif:
        # Concatenate samples from original Train and Test sets
        # Only 'training' physionet samples are have labels.
        # Therefore, if we do classifiction task, we don't need physionet 'test' samples.
        total_dataset = total_dataset + \
            test_dataset_obj[:len(test_dataset_obj)]
    print(len(total_dataset))
    # Shuffle and split
    train_data, test_data = model_selection.train_test_split(total_dataset, train_size=0.8,
                                                             random_state=42, shuffle=True)

    record_id, tt, vals, mask, labels = train_data[0]

    # n_samples = len(total_dataset)
    input_dim = vals.size(-1)
    data_min, data_max = get_data_min_max(total_dataset, device)
    batch_size = min(min(len(train_dataset_obj), args.batch_size), args.n)

    def extrap(test_data):
        enc_test_data = []
        dec_test_data = []
        for (record_id, tt, vals, mask, labels) in test_data:
            midpt = 0
            for tp in tt:
                if tp < 24:
                    midpt += 1
                else:
                    break
            if mask[:midpt].sum() and mask[midpt:].sum():
                enc_test_data.append(
                    (record_id, tt[:midpt], vals[:midpt], mask[:midpt], labels))
                dec_test_data.append(
                    (record_id, tt[midpt:], vals[midpt:], mask[midpt:], labels))
        return enc_test_data, dec_test_data

    enc_train_data, dec_train_data = extrap(train_data)
    enc_test_data, dec_test_data = extrap(test_data)
    enc_train_data_combined = variable_time_collate_fn(
        enc_train_data, device, classify=args.classif, data_min=data_min, data_max=data_max)
    dec_train_data_combined = variable_time_collate_fn(
        dec_train_data, device, classify=args.classif, data_min=data_min, data_max=data_max)
    enc_test_data_combined = variable_time_collate_fn(
        enc_test_data, device, classify=args.classif, data_min=data_min, data_max=data_max)
    dec_test_data_combined = variable_time_collate_fn(
        dec_test_data, device, classify=args.classif, data_min=data_min, data_max=data_max)
    print(enc_train_data_combined.shape, dec_train_data_combined.shape)
    print(enc_test_data_combined.shape, dec_test_data_combined.shape)

    # keep the timepoints in enc between 0.0 and 0.5
    enc_train_data_combined[:, :, -1] *= 0.5
    enc_test_data_combined[:, :, -1] *= 0.5
    print(enc_train_data_combined[0, :, -1], dec_train_data_combined[0, :, -1])
    enc_train_dataloader = DataLoader(
        enc_train_data_combined, batch_size=batch_size, shuffle=False)
    dec_train_dataloader = DataLoader(
        dec_train_data_combined, batch_size=batch_size, shuffle=False)
    enc_test_dataloader = DataLoader(
        enc_test_data_combined, batch_size=batch_size, shuffle=False)
    dec_test_dataloader = DataLoader(
        dec_test_data_combined, batch_size=batch_size, shuffle=False)

    attr_names = train_dataset_obj.params
    data_objects = {"dataset_obj": train_dataset_obj,
                    "enc_train_dataloader": enc_train_dataloader,
                    "enc_test_dataloader": enc_test_dataloader,
                    "dec_train_dataloader": dec_train_dataloader,
                    "dec_test_dataloader": dec_test_dataloader,
                    "input_dim": input_dim,
                    "attr": attr_names,  # optional
                    "classif_per_tp": False,  # optional
                    "n_labels": 1}  # optional

    return data_objects


def subsample_timepoints(data, time_steps, mask, percentage_tp_to_sample=None):
    # Subsample percentage of points from each time series
    for i in range(data.size(0)):
        # take mask for current training sample and sum over all features --
        # figure out which time points don't have any measurements at all in this batch
        current_mask = mask[i].sum(-1).cpu()
        non_missing_tp = np.where(current_mask > 0)[0]
        n_tp_current = len(non_missing_tp)
        n_to_sample = int(n_tp_current * percentage_tp_to_sample)
        subsampled_idx = sorted(np.random.choice(
            non_missing_tp, n_to_sample, replace=False))
        tp_to_set_to_zero = np.setdiff1d(non_missing_tp, subsampled_idx)

        data[i, tp_to_set_to_zero] = 0.
        if mask is not None:
            mask[i, tp_to_set_to_zero] = 0.

    return data, time_steps, mask

import numpy as np
import torch
import os
import hashlib
import torch.distributed as dist
from helper_functions import *

world_size = dist.get_world_size()
local_rank = dist.get_rank()


def deterministic_random(id_value, seed_value):
    tensor_bytes = id_value.detach().cpu().numpy().tobytes()
    seed_bytes = str(seed_value).encode("utf-8")
    data_to_hash = tensor_bytes + seed_bytes
    hash_object = hashlib.sha256(data_to_hash)
    hex_digest = hash_object.hexdigest()
    seed = int(hex_digest[0:16], 16)

    g = torch.Generator(device=f"cuda:{local_rank}")
    g.manual_seed(seed)
    return g


from tqdm import tqdm
from scipy import linalg

from torch.utils.data import DistributedSampler, DataLoader

from utils.metrics import *
from models import predict_model_output
from datasets import get_initial_noise, repeated_samples

def generate_id_samples_from_diff(
    model,
    noise_scheduler,
    config,
    samples_to_pass_model,
    fake_embeddings,
    generators_to_pass_model=None,
):
    for timesteps in noise_scheduler.timesteps:
        noise_pred = predict_model_output(
            model,
            samples_to_pass_model,
            timesteps,
        )

        samples_to_pass_model = noise_scheduler.step(
            noise_pred,
            timesteps,
            samples_to_pass_model,
            generator=generators_to_pass_model,
        ).prev_sample

    fake_embeddings.append(samples_to_pass_model.squeeze(1).cpu().numpy())


def generate_ids(
    num_of_samples_to_generate_in_one_model_pass,
    model,
    dataset,
    noise_scheduler,
    fake_embeddings,
    count_per_id,
    config,
    **kwargs,
):
    samples = get_initial_noise(num_of_samples_to_generate_in_one_model_pass, config)

    # this if block is mostly for debug purposes
    if count_per_id > 1:
        # Iterating over ids to collect samples per id where total sample count doesn't exceed the batch_size
        collected_samples = []
        generators_to_pass_model = []
        for e, id_sample in enumerate(samples):
            # repeat the same initial noise vector for the same id 50 times
            same_ids_samples = repeated_samples(id_sample, count_per_id, config)

            collected_samples.append(same_ids_samples)

            # if we are not using config.model.use_deterministic_randomness, this gets ignored anyways
            generators_to_pass_model.append(
                [
                    (
                        deterministic_random(sample, (kwargs["BASE_SEED"] + local_rank))
                        if config.model.use_deterministic_randomness
                        else None
                    )
                    for sample in same_ids_samples
                ]
            )

            """
                1- iterate over ids
                2- populate the batch by placing a total of count_per_id samples for each iterated id
                3- stop when collected number of samples is about to go over the batch_size
            """
            if (e + 2) * count_per_id < num_of_samples_to_generate_in_one_model_pass:
                continue

            samples_to_pass_model = torch.cat(collected_samples).unsqueeze(1)

            generators_to_pass_model = (
                [gen for gen_list in generators_to_pass_model for gen in gen_list]
                if config.model.use_deterministic_randomness
                else None
            )
            generate_id_samples_from_diff(
                model,
                noise_scheduler,
                config,
                samples_to_pass_model,
                fake_embeddings,
                generators_to_pass_model,
            )

            # empty out the collected values
            collected_samples = []
            generators_to_pass_model = []
    else:
        generate_id_samples_from_diff(
            model,
            noise_scheduler,
            config,
            samples,
            fake_embeddings,
        )


def evaluate(model, noise_scheduler, dataset, config, **kwargs):
    model.eval()

    count_per_id = config.eval.count_per_id
    batch_size = config.training.batch_size
    num_generated = config.eval.num_generated
    num_per_rank = num_generated // world_size
    if local_rank < num_generated % world_size:
        num_per_rank += 1

    # generate ids
    fake_embeddings = []
    with torch.no_grad():
        for _ in tqdm(range(num_per_rank // batch_size), desc="Iterating over IDs"):
            generate_ids(
                batch_size,
                model,
                dataset,
                noise_scheduler,
                fake_embeddings,
                count_per_id,
                config,
                **kwargs,
            )

        # continue to generate to end up with exactly num_per_rank ids
        if num_per_rank % batch_size != 0:
            generate_ids(
                num_per_rank % batch_size,
                model,
                dataset,
                noise_scheduler,
                fake_embeddings,
                count_per_id,
                config,
                **kwargs,
            )

    # concat, reshape and normalize
    fake_embeddings = np.concatenate(fake_embeddings, axis=0)
    fake_embeddings = fake_embeddings.reshape(fake_embeddings.shape[0], -1)
    fake_embeddings = fake_embeddings / np.linalg.norm(
        fake_embeddings, axis=1, keepdims=True
    )

    if count_per_id > 1:
        fake_embeddings = fake_embeddings.reshape(num_per_rank, count_per_id, -1)

    if config.eval.calculate_metrics:
        # only do this population part if config.real_embeddings is empty at first time
        if len(config.real_embeddings) == 0:
            r_emb = []
            # THIS TWO LINE IS FOR DISTRIBUTED FID CALCULATION
            # sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=local_rank)
            # dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

            # EACH GPU GENERATES A PORTION OF THE EMBEDDINGS AND CALCULATES THE METRICS ON IT
            dataloader = DataLoader(dataset, batch_size=batch_size)

            # collect, concat, reshape and normalize
            for batch in dataloader:
                vec = batch
                r_emb.append(vec.numpy())
            r_emb = np.concatenate(r_emb, axis=0)
            r_emb = r_emb.reshape(r_emb.shape[0], -1)
            r_emb = r_emb / np.linalg.norm(r_emb, axis=1, keepdims=True)

            config.real_embeddings = r_emb

        # Compute metrics
        fid_score = calculate_fid(config.real_embeddings, fake_embeddings)
        uniqueness_score, min_similarity, max_similarity = calculate_uniqueness(
            fake_embeddings
        )

        same_id_avg_sim, same_id_min_sim, same_id_max_sim = (
            calculate_intra_class_similarity(fake_embeddings)
        )
    else:
        print("Metric calculation skipped")
        fid_score, uniqueness_score = 0, 0
        min_similarity, max_similarity = 0, 0
        same_id_avg_sim, same_id_min_sim, same_id_max_sim = 0, 0, 0

    print(
        f"RANK[{local_rank}] --> FID: {fid_score}, Uniqueness: {uniqueness_score}, Min/Max Similarity: ({min_similarity}, {max_similarity}), Same ID sim: {same_id_avg_sim}, Same ID min/max sim: ({same_id_min_sim}, {same_id_max_sim})"
    )

    return (
        fake_embeddings,
        fid_score,
        uniqueness_score,
        (min_similarity, max_similarity),
        same_id_avg_sim,
        (same_id_min_sim, same_id_max_sim),
    )

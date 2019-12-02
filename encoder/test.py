from encoder.data_objects import SpeakerVerificationDataLoader, SpeakerVerificationDataset
from encoder.params_model import *
from encoder.model import SpeakerEncoder
from utils.profiler import Profiler
from pathlib import Path
from tqdm import tqdm
import torch
import csv
import os

def sync(device: torch.device):
    # FIXME
    return 
    # For correct profiling (cuda operations are async)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

def cosine_similarity(new_embeds, baseline_embeds):
    """Computes the cosine similarity for given embeddings.

    Given two embeddings of dimension [embedding_size] x [speaker_batch] x [utterances_per_speaker],
    calculates the cosine similarity and returns a 1d tensor of size speaker_batch x 
    utterances_per_speaker.

    Returns: cosine similarity list for all partial utterances in batch.
    """
    flat_new_embeds = torch.flatten(new_embeds, start_dim=0, end_dim=1)
    flat_baseline_embeds = torch.flatten(baseline_embeds, start_dim=0, end_dim=1)
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    cos_similarity = cos(flat_new_embeds, flat_baseline_embeds)
    return cos_similarity

def test_all(run_id_baseline: str, test_data_root: Path, models_dir: Path):
    model_paths = list(models_dir.glob("*.pt"))
    test_paths = list(test_data_root.glob("*"))
    for model in model_paths:
        # Remove .pt from name.
        model_id = os.path.basename(model)[:-3]
        for test_path in test_paths:
            preprocessed_dir = Path(os.path.join(test_path, "SV2TTS_test/encoder"))
            print("Testing model %s on dir %s." % (model_id, preprocessed_dir))
            test(model_id, run_id_baseline, preprocessed_dir, models_dir, False)



def test(run_id: str, run_id_baseline: str, test_data_root: Path, models_dir: Path, test_full: bool):
    if test_full:
        test_all(run_id_baseline, test_data_root, models_dir)
        return
    with torch.no_grad():
        # Create a dataset and a dataloader
        dataset = SpeakerVerificationDataset(test_data_root)
        loader = SpeakerVerificationDataLoader(
            dataset,
            speakers_per_batch,
            utterances_per_speaker,
            num_workers=8,
        )
        
        # Setup the device on which to run the forward pass and the loss. These can be different, 
        # because the forward pass is faster on the GPU whereas the loss is often (depending on your
        # hyperparameters) faster on the CPU.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # FIXME: currently, the gradient is None if loss_device is cuda
        loss_device = torch.device("cpu")
        
        # Create the model
        model = SpeakerEncoder(device, device)
        
        # Configure file path for the model
        state_fpath = models_dir.joinpath(run_id + ".pt")

        # Load the baseline model if available.
        if run_id_baseline:
            model_baseline = SpeakerEncoder(device, device)
            baseline_state_fpath = models_dir.joinpath(run_id_baseline + ".pt")
            if baseline_state_fpath.exists():
                print("Found baseline model \"%s\", loading it." % run_id_baseline)
                checkpoint_baseline = torch.load(baseline_state_fpath)
                model_baseline.load_state_dict(checkpoint_baseline["model_state"])
                model_baseline.eval()
            else:
                print("No model \"%s\" found." % run_id_baseline)

        # Load the model
        if state_fpath.exists():
            print("Found model \"%s\", loading it." % run_id)
            checkpoint = torch.load(state_fpath)
            model.load_state_dict(checkpoint["model_state"])
        else:
            print("No model \"%s\" found." % run_id)
            return
        model.eval()
        
        # Testing loop
        counter = 0
        eer_total = 0
        eer_baseline_total = 0.0
        cosine_similarity_total = 0.0
        for idx, speaker_batch in enumerate(loader):
            
            # Forward pass
            inputs = torch.from_numpy(speaker_batch.data).to(device)
            sync(device)
            embeds = model(inputs)
            sync(device)
            embeds_loss = embeds.view((speakers_per_batch, utterances_per_speaker, -1)).to(device)
            loss, eer = model.loss(embeds_loss)
            sync(device)
            eer_total += eer
            
            # If we added a baseline, compute cosine similarity too.
            if (model_baseline):
                baseline_embeds = model_baseline(inputs)
                sync(device)
                embeds_loss_baseline = baseline_embeds.view((speakers_per_batch, utterances_per_speaker, -1)).to(device)
                loss, eer = model.loss(embeds_loss_baseline)
                sync(device)
                eer_baseline_total += eer
                batch_similarity = cosine_similarity(embeds_loss, embeds_loss_baseline)
                batch_similarity_sum = torch.sum(batch_similarity)
                batch_similarity_sum = (batch_similarity_sum / batch_similarity.shape[0])
                batch_similarity_sum = batch_similarity_sum.to(loss_device)
                cosine_similarity_total += batch_similarity_sum

            counter += 1
            if idx % 100 == 0:
                print("Batch %d/%d done." % (idx, len(loader) / (speakers_per_batch * utterances_per_speaker)))
            if idx == 1000 or idx == (len(loader)/5):
                break

        csv_path = os.path.join(models_dir, "results.csv")
        print("EER for model \"%s\" = %.4f" % (run_id, eer_total / counter))
        if run_id_baseline:
            print("EER for model \"%s\" = %.4f" % (run_id_baseline, eer_baseline_total / counter))
            print("Cosine similarity between both = %.4f" % (cosine_similarity_total / counter))

        result_list = []
        result_list.append([run_id, test_data_root, eer_total / counter, float(cosine_similarity_total / counter)])


        # Write results.
        if os.path.exists(csv_path):
            with open(csv_path, "a") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerows(result_list)
        else:
            with open(csv_path, "w") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerows(result_list)
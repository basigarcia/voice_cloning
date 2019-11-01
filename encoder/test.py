from encoder.data_objects import SpeakerVerificationDataLoader, SpeakerVerificationDataset
from encoder.params_model import *
from encoder.model import SpeakerEncoder
from utils.profiler import Profiler
from pathlib import Path
from tqdm import tqdm
import torch

def sync(device: torch.device):
    # FIXME
    return 
    # For correct profiling (cuda operations are async)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

def test(run_id: str, test_data_root: Path, models_dir: Path):
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
    model = SpeakerEncoder(device, loss_device)
    
    # Configure file path for the model
    state_fpath = models_dir.joinpath(run_id + ".pt")

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
    for speaker_batch in loader:
        
        # Forward pass
        inputs = torch.from_numpy(speaker_batch.data).to(device)
        sync(device)
        embeds = model(inputs)
        sync(device)
        embeds_loss = embeds.view((speakers_per_batch, utterances_per_speaker, -1)).to(loss_device)
        loss, eer = model.loss(embeds_loss)
        sync(loss_device)
        eer_total += eer
        counter += 1
        if counter >= 150:
            break
    
    print(eer_total / counter)
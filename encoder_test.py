from utils.argutils import print_args
from encoder.test import test
from pathlib import Path
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test models against the test dataset. You must have trained or downloaded models first.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("run_id", type=str, help= \
        "Name for this model instance. It must be a already trained model.")
    parser.add_argument("run_id_baseline", type=str, help= \
        "Name for the baseline model to compare the new one against for cosine similarity.")
    parser.add_argument("test_data_root", type=Path, help= \
        "Path to the output directory of encoder_preprocess.py. If you left the default "
        "output directory when preprocessing, it should be <datasets_root>/SV2TTS/encoder/.")
    parser.add_argument("-m", "--models_dir", type=Path, default="encoder/saved_models/", help=\
        "Path to the directory that will contain the saved model weights.")
    # parser.add_argument("compute_similarity", type=bool, help= \
    #     "Wheter to compute cosine similarity between the embeddings of the pairs of models in models_dir.")
    args = parser.parse_args()
    
    # Process the arguments
    args.models_dir.mkdir(exist_ok=True)
    
    # Run the testing.
    print_args(args, parser)
    test(**vars(args))
    
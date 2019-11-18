from IPython.display import Audio
from IPython.utils import io
from encoder.test import cosine_similarity
from synthesizer.inference import Synthesizer
from encoder import inference as encoder
from vocoder import inference as vocoder
from pathlib import Path
from toolbox import Toolbox
from utils.argutils import print_args
import numpy as np
import librosa
import argparse
import os
import scipy.spatial.distance as distance
import csv

def synth_and_eval(test_encoder, vocoder, synthesizer, input_wav, base_encoder):
  text = "This is being said in my own voice.  The computer has learned to do an impression of me." #@param {type:"string"}
  in_fpath = Path(input_wav)
  # reprocessed_wav = encoder.preprocess_wav(in_fpath)
  original_wav, sampling_rate = librosa.load(in_fpath)
  preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
  embed = encoder.embed_utterance(preprocessed_wav, model=test_encoder)
  base_embed = encoder.embed_utterance(preprocessed_wav, model=base_encoder)
  print("\nSynthesizing new audio...")
  with io.capture_output() as captured:
    specs = synthesizer.synthesize_spectrograms([text], [embed])
  generated_wav = vocoder.infer_waveform(specs[0])
  generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")

  # Generate embedding for synthesized utterance (with pretrained encoder).
  preprocessed_gen_wav = encoder.preprocess_wav(generated_wav, sampling_rate)
  synth_embed = encoder.embed_utterance(preprocessed_gen_wav, model=base_encoder)

  # Compare original and synthesized embeddings.
  cosine_similarity = distance.cosine(base_embed, synth_embed)

  return generated_wav, synthesizer.sample_rate, cosine_similarity

def TestFullSystem(datasets_root, enc_model_dir, syn_model_dir, voc_model_dir, low_mem, base_enc_model_dir):
	"""Tests a model end-to-end with a given small dataset.
	Loads the encoder, vocoder, and synthesizers and runs a set of utterances through them
	to generate synthesized versions of those utterances. Then computes the cosine
	similarity between the original/synthesized pairs and outputs the results to a
	csv file.

	Synthesizing utterances is slow, so it takes aboug 4x input length in time to run. It
	is recommended to only use this with small sets of a handful of utterances at a time.
	"""
	encoder.set_device()
	encoder_weights = Path(enc_model_dir)
	base_encoder_weights = Path(base_enc_model_dir)
	vocoder_weights = Path(voc_model_dir)
	base_encoder = encoder.get_model(base_encoder_weights)
	test_encoder = encoder.get_model(encoder_weights)
	synthesizer = Synthesizer(syn_model_dir)
	vocoder.load_model(vocoder_weights)

	encoder_name = os.path.splitext(os.path.basename(enc_model_dir))[0]
	testset_name = os.path.basename(datasets_root)
	dataset_name = testset_name + "_synth"
	dataset_dir = Path(os.path.join(os.path.dirname(datasets_root), dataset_name))
	dataset_dir.mkdir(exist_ok=True)

	running_cos_similarity = 0.0
	running_count = 0
	result_list = []
	csv_path = os.path.join(dataset_dir, "results.csv")
	if not os.path.exists(csv_path):
		result_list.append(["Model", "Testset", "Speaker", "Audio File", "Audio Duration (sec)", "Cosine Similarity"])
	# Load utterances from testset.
	speaker_dirs = list(datasets_root.glob("*"))
	for speaker_dir in speaker_dirs:
		audio_file = list(speaker_dir.glob("*"))[0]
		file_name = os.path.basename(audio_file)
		speaker_id = os.path.basename(speaker_dir)
		output_name = os.path.splitext(file_name)[0] + "_synth_" + encoder_name + ".wav"
		output_path = Path(os.path.join(dataset_dir, speaker_id))
		output_path.mkdir(exist_ok=True)
		output_path = os.path.join(output_path, output_name)

		synthesized_wav, sampling_rate, similarity = synth_and_eval(
			test_encoder, vocoder, synthesizer, audio_file, base_encoder)
		librosa.output.write_wav(output_path, synthesized_wav.astype(np.float32), sampling_rate)
		running_cos_similarity += similarity
		running_count += 1
		result_list.append(
			[encoder_name, testset_name, speaker_id, file_name, 
			(len(synthesized_wav)/sampling_rate), similarity])
	result_list.append(
		[encoder_name, testset_name, "ALL", "ALL", "ALL", running_cos_similarity / running_count])

	# Write results.
	if os.path.exists(csv_path):
		with open(csv_path, "a") as csv_file:
			writer = csv.writer(csv_file)
			writer.writerows(result_list)
	else:
		with open(csv_path, "w") as csv_file:
			writer = csv.writer(csv_file)
			writer.writerows(result_list)

	print("\n-------Overall cosine similarity-------")
	print("%.4f" % (running_cos_similarity / running_count))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Tests the entire system.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("-d", "--datasets_root", type=Path, help= \
        "Path to the datasets. i.e. ../../Datasets/FullTest/VoxCeleb2",
                        default=None)
    parser.add_argument("-e", "--enc_model_dir", type=Path, default="encoder/saved_models/pretrained.pt", 
                        help="Directory containing saved encoder models")
    parser.add_argument("-s", "--syn_model_dir", type=Path, default="synthesizer/saved_models/logs-pretrained/taco_pretrained", 
                        help="Directory containing saved synthesizer models")
    parser.add_argument("-v", "--voc_model_dir", type=Path, default="vocoder/saved_models/pretrained/pretrained.pt", 
                        help="Directory containing saved vocoder models")
    parser.add_argument("--low_mem", action="store_true", help=\
        "If True, the memory used by the synthesizer will be freed after each use. Adds large "
        "overhead but allows to save some GPU memory for lower-end GPUs.")
    parser.add_argument("-b", "--base_enc_model_dir", type=Path, default="encoder/saved_models/pretrained.pt", 
    	help="Directory containing saved encoder models")
    args = parser.parse_args()

    # Launch the toolbox
    print_args(args, parser)
    TestFullSystem(**vars(args))

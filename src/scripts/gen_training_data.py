# from transformers import pipeline
import h5py
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

# Set the desired source_dir and filename here.
source_dir = 'data/ptb/'
filename = 'ptb_train'

source_file = source_dir + filename + '.txt'
targ_file = source_dir + filename + '.hdf5'

file1 = open(source_file, 'r')
idx = 0
hf = h5py.File(targ_file, 'w')
for line in file1:
    print()
    print("Analyzing line number", idx)
    if '?' in line:  # Question case:
        q_mark_idx = line.index('?')
        context = line[q_mark_idx + 2:]
        question = line[:q_mark_idx + 1]
        print("Context", context)
        print("Question", question)
        inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt")
    else:
        inputs = tokenizer.encode_plus(tokenizer.wordpiece_tokenizer.tokenize(line), return_tensors='pt')
    with torch.no_grad():
        model_output = model(**inputs, output_hidden_states=True)
        answer_start = torch.argmax(model_output[0], dim=1).numpy()[0]
        answer_end = (torch.argmax(model_output[1], dim=1) + 1).numpy()[0]
        selected_tokens = inputs["input_ids"][0][answer_start:answer_end]
        print("Answer", tokenizer.decode(selected_tokens))
        _, _, embeddings = model_output
        np_embeddings = np.zeros((25, embeddings[0].shape[1], embeddings[0].shape[2]))
        for layer in range(25):
            np_embeddings[layer] = embeddings[layer].numpy()
        # For each of the 25 layers in the model, I have a 1 x s_length x 768 tensor.
        # Now write the embeddings to an hdf5 file for training a probe with later.
        hf.create_dataset(str(idx), data=np_embeddings)
    idx += 1
    if idx > 5000:
        print("Breaking after 5000; the only reason to have that many examples is if you are generating data to train a probe.")
        break
file1.close()
hf.close()

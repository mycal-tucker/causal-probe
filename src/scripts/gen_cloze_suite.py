import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM

"""
Creates a suite of masked sentences like 

The man saw the boy and the dog [MASK] tired.

So, using "was" vs. "were" is ambiguous.
"""

tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking")
model = AutoModelForMaskedLM.from_pretrained("bert-large-uncased-whole-word-masking")


def create_sentence():
    sentence_list = ["The", n1, v1, "the", n2, 'and the', n3, '[MASK]', adj + '.']
    sentence = " ".join(sentence_list)
    return sentence


def create_tree():
    tree1 = '(ROOT\n' +\
                '(S\n' +\
                    '(NP (DT The) (NN ' + n1 + '))\n' +\
                    '(VP (VBD ' + v1 + ')\n' +\
                        '(SBAR\n' +\
                            '(S\n' +\
                                '(NP\n' +\
                                    '(NP (DT the) (NN ' + n2 + '))\n' +\
                                    '(CC and)\n' +\
                                    '(NP (DT the) (NN ' + n3 + ')))\n' +\
                                '(VP (VBD [MASK])\n' +\
                                    '(ADJP (JJ ' + adj + '))))))\n' +\
             '(. .)))\n\n'
    tree2 = '(ROOT\n' +\
              '(S\n' +\
                    '(S\n' +\
                         '(NP (DT The) (NN ' + n1 + '))\n' +\
                            '(VP (VBD ' + v1 + ')\n' +\
                                '(NP (DT the) (NN ' + n2 + '))))\n' +\
                    '(CC and)\n' +\
                    '(S\n' +\
                          '(NP (DT the) (NN ' + n3 +'))\n' +\
                            '(VP (VBD [MASK])\n' +\
                                '(ADJP (JJ ' + adj + '))))\n' +\
                    '(. .)))\n\n'
    return tree1, tree2


candidates = set()

fn_map = {'cloze': (create_sentence, create_tree)}  # If you want to add indirection, can have multiple templates or something.

root_dir = 'data/example/'
nn1 = ['man', 'woman', 'child']
nn2 = ['boy', 'building', 'cat']
nn3 = ['dog', 'girl', 'truck']
v1s = ['saw', 'feared', 'heard']
adjs = ['tall', 'falling', 'orange']
mask_id = tokenizer.convert_tokens_to_ids("[MASK]")

line_idx = 0
for corpus_type, fns in fn_map.items():
    sentence_fn, tree_fn = fns
    for n1 in nn1:
        for n2_idx, n2 in enumerate(nn2):
            for n3_idx, n3 in enumerate(nn3):
                if n2 == n3:
                    continue
                for v1 in v1s:
                    for adj in adjs:
                        text = sentence_fn()
                        np_text = tokenizer(text, return_tensors='np')
                        mask_idx = np.where(np_text.data['input_ids'] == mask_id)[1][0]

                        tokenized_text = tokenizer.encode_plus(tokenizer.wordpiece_tokenizer.tokenize(text),
                                                               return_tensors='pt')
                        prediction, hidden_states = model(**tokenized_text, output_hidden_states=True)

                        np_prediction = prediction.cpu().detach().numpy()
                        overall_best = np.argmax(np_prediction[0, mask_idx])
                        best_token = tokenizer.ids_to_tokens.get(overall_best)
                        print(text, "\t", best_token)
                        candidates.add(best_token)

                        parses = tree_fn()
                        for parse in parses:
                            # Dump data into text files
                            file_mode = 'w' if line_idx == 0 else 'a'
                            with open(root_dir + 'text.txt', file_mode) as text_file:
                                text_file.write(text + '\n')
                            # Dump information about the query into a file for future analysis.
                            # This includes the location of the masked word, plus a list of predicted words across
                            # the whole corpus. In general, we pipe extra information like this through this file.
                            with open(root_dir + 'token_idxs.txt', file_mode) as token_idx_file:
                                token_idx_file.write('\t'.join([str(entry) for entry in [corpus_type, mask_idx] + list(candidates)]) + '\n')
                            # And write the tree itself.
                            with open(root_dir + 'text.trees', file_mode) as tree_file:
                                tree_file.write(parse)
                            line_idx += 1

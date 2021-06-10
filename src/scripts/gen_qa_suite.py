import numpy as np
import torch
from scipy.special import softmax
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")


def logit_to_prob(logit):
    return softmax(logit)


def gen_trees():
    # Only n1 modified is first one.
    subject_template1 = '(ROOT\n' +\
        '(S\n' +\
        '(NP ' + question_word + ')\n' +\
        '(NP had)\n' +\
        '(NP the)\n' +\
        '(NP ' + n3 + ')\n' +\
        '(NP ?)\n' +\
        '(NP (DT The) (NN ' + n1 + '))\n' +\
            '(VP (VBD ' + v + ')\n' +\
            '(NP (DT the) (NN ' + n2 + '))\n' +\
            '(PP (IN with)\n' +\
                '(NP (DT the) (NN ' + n3 + '))))\n' +\
        '(. .)))\n\n'
    # Only n2 modified is second one.
    subject_template2 = '(ROOT\n' +\
        '(S\n' +\
        '(NP ' + question_word + ')\n' +\
        '(NP had)\n' +\
        '(NP the)\n' +\
        '(NP ' + n3 + ')\n' +\
        '(NP ?)\n' +\
        '(NP (DT The) (NN ' + n1 + '))\n' +\
            '(VP (VBD ' + v + ')\n' +\
            '(NP \n' +\
                '(NP (DT the) (NN ' + n2 + '))\n' +\
                '(PP (IN with)\n' +\
                    '(NP (DT the) (NN ' + n3 + ')))))\n' +\
        '(. .)))\n\n'
    return subject_template1, subject_template2


def gen_sentence():
    c = " ".join(["The", n1, v, "the", n2, "with the", n3 + '.'])
    q = question_word + " had the " + n3 + "?"
    return c, q


def get_token_idx(ids, str):
    for token_idx, input_id in enumerate(ids):
        string_for_id = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_id))
        stripped_str = "".join(string_for_id.split(" "))  # Gets rid of all whitespaces
        if stripped_str == str:
            print("Found string", str, "at idx", token_idx)
            return token_idx
    assert False, "Could not find " + str + " from input ids."


fn_map = {'np_vp': (gen_sentence, gen_trees)}

root_dir = 'data/qa_example/'
corpus_types = ['np_vp']

nn1 = ['man', 'woman', 'child']
nn2 = ['man', 'woman', 'boy', 'girl', 'stranger', 'dog']
others = [('saw', 'telescope'), ('poked', 'stick'), ('thanked', 'letter'), ('fought', 'knife'), ('dressed', 'hat'), ('indicated', 'ruler'), ('kicked', 'shoe'), ('welcomed', 'gift'), ('buried', 'shovel')]
question_word = 'Who'
rel_clause_word = 'with'
# Build all the candidate inputs.
line_idx = 0
for corpus_type in corpus_types:
    print("Corpus type", corpus_type)
    sentence_fn, tree_fn = fn_map.get(corpus_type)
    for n1_idx, n1 in enumerate(nn1):
        for n2_idx, n2 in enumerate(nn2):
            if n1 == n2:
                continue
            for other in others:
                v, n3 = other
                context, question = sentence_fn()
                templates = tree_fn()
                for template_idx, template in enumerate(templates):
                    # Tokenize the sentence and create tensor inputs to BERT
                    inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt")
                    input_ids = inputs["input_ids"].tolist()[0]
                    nn1_idx = get_token_idx(input_ids, n1)
                    det1_idx = nn1_idx - 1 if corpus_type != 'subject_modifiers' else nn1_idx - 2
                    nn2_idx = get_token_idx(input_ids, n2)
                    det2_idx = nn2_idx - 1

                    answer_start_scores, answer_end_scores = model(**inputs)
                    answer_start = torch.argmax(
                        answer_start_scores
                    )  # Get the most likely beginning of answer with the argmax of the score
                    start_likelihood = np.max(logit_to_prob(answer_start_scores.detach().numpy()))
                    answer_end = torch.argmax(
                        answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score
                    answer = tokenizer.convert_tokens_to_string(
                        tokenizer.convert_ids_to_tokens(input_ids[answer_start: answer_end]))
                    print()
                    print("Analyzing the context", context)
                    print(f"Question: {question}")
                    print(f"Answer: {answer}")
                    print("Start likelihood", start_likelihood)
                    answer_length = len(answer.split(' '))
                    print("Answer length", answer_length)
                    # Dump data into text files
                    file_mode = 'w' if line_idx == 0 else 'a'
                    with open(root_dir + 'text.txt', file_mode) as text_file:
                        text_file.write(question + ' ' + context + '\n')
                    # Where were the various tokens?
                    with open(root_dir + 'token_idxs.txt', file_mode) as token_idx_file:
                        token_idx_file.write('\t'.join([str(entry) for entry in [det1_idx, nn1_idx, det2_idx, nn2_idx]]) + '\n')
                    # Lots of info about the question and answer
                    with open(root_dir + 'setup.txt', file_mode) as question_file:
                        question_file.write('\t'.join([corpus_type, str(answer_length), str(start_likelihood), context, question, answer]) + '\n')
                    # And write the tree itself.
                    with open(root_dir + 'text.trees', file_mode) as tree_file:
                        tree_file.write(template)
                    line_idx += 1

import json
from argparse import ArgumentParser
from tqdm import tqdm
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from transformers.pipelines.pt_utils import KeyDataset
from datasets import load_dataset
from datasets import concatenate_datasets
import evaluate
import nltk
import numpy as np
import random
from nltk.tokenize import sent_tokenize
from transformers import DataCollatorForSeq2Seq, pipeline
from sklearn.metrics import f1_score, accuracy_score
from typing import DefaultDict
from eval_helpers import calculate_f1, precision, recall
import time
import torch

nltk.download("punkt")

parser = ArgumentParser(description='Arguments for training')
parser.add_argument('--dataset', type=str, help='dataset name', default="multi3nlu")
parser.add_argument('--fold', type=int, help='Fold', default=0)
parser.add_argument('--template_name', type=str, help='Template key', default="none_none_none")
parser.add_argument('--evaluate', action="store_true", help='Whether to evaluate model')
parser.add_argument('--large', action="store_true", help='Whether to use T5 large model')
parser.add_argument('--small', action="store_true", help='Whether to use T5 small model')
parser.add_argument('--xlarge', action="store_true", help='Whether to use T5 XL model')
parser.add_argument('--model_type', type=str, help='Model type -- tuned for QA/instructions', default="instr")
parser.add_argument('--model_name', type=str, help='Model name', default="models/flan_base_0/")
parser.add_argument('--language', type=str, help='language', default="english")
parser.add_argument('--domain', type=str, help='Domain', default="banking")
parser.add_argument('--setting', type=int, help='Data setting [Options: 1, 10, 20]', default=10)
parser.add_argument('--task', type=str, help='Task to work on [slots, intents]', default="intents")
parser.add_argument('--data_filter', type=str, help='How to filter the data: by folds/random', default="folds")
parser.add_argument('--num_examples', type=int, help='Number of random examples', default=500)
args = parser.parse_args()

if args.language=="english":
    if args.data_filter == "folds":
        train_file = os.path.join("..", args.dataset, "english", f"train_{args.fold}_{args.template_name}_{args.setting}_{args.domain}_{args.task}.json")
        test_file = os.path.join("..", args.dataset, "english", f"test_{args.fold}_{args.template_name}_{args.setting}_{args.domain}_{args.task}.json")
    else:
        train_file = os.path.join("..", args.dataset, "english", f"train_random_{args.fold}_{args.num_examples}_{args.template_name}_{args.setting}_{args.domain}_{args.task}.json")
        test_file = os.path.join("..", args.dataset, "english", f"test_random_{args.fold}_{args.num_examples}_{args.template_name}_{args.setting}_{args.domain}_{args.task}.json")
else:
    train_file = os.path.join("..", args.dataset, args.language, f"train_{args.fold}_{args.template_name}_{args.setting}_{args.domain}_{args.task}.json")
    test_file = os.path.join("..", args.dataset, args.language, f"test_{args.fold}_{args.template_name}_{args.setting}_{args.domain}_{args.task}.json")

print(train_file)
print(test_file)
if not args.evaluate:
    if args.model_type == "instr":
        if args.large:
            model_id="google/flan-t5-large"
        elif args.small:
           model_id="google/flan-t5-small"
        elif args.xlarge:
           model_id="google/flan-t5-xl"
        else:
            model_id="google/flan-t5-base"
    else:
        model_id = "mrm8488/t5-base-finetuned-squadv2"
else:
    model_id = args.model_name
# Load tokenizer of FLAN-t5-base
if args.model_type=="instr":
    if args.large:
        tokenizer_id="google/flan-t5-large"
    elif args.small:
        tokenizer_id="google/flan-t5-small"
    elif args.xlarge:
        tokenizer_id="google/flan-t5-xl"
    else:
        tokenizer_id="google/flan-t5-base"
else:
    tokenizer_id="mrm8488/t5-base-finetuned-squadv2"
if "t5" not in model_id and "flan" not in model_id:
    tokenizer_id = model_id

tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset("json", data_files={"train":train_file, "test":test_file})
print("LOADED")

tokenized_inputs = concatenate_datasets([dataset["train"], dataset["test"]]).map(lambda x: tokenizer(x["input"], truncation=True), batched=True, remove_columns=["input", "labels"])
max_source_length = max([len(x) for x in tokenized_inputs["input_ids"]])
print(f"Max source length: {max_source_length}")

# The maximum total sequence length for target text after tokenization.
# Sequences longer than this will be truncated, sequences shorter will be padded."
tokenized_targets = concatenate_datasets([dataset["train"], dataset["test"]]).map(lambda x: tokenizer(x["labels"], truncation=True), batched=True, remove_columns=["input", "labels"])
max_target_length = max([len(x) for x in tokenized_targets["input_ids"]])
print(f"Max target length: {max_target_length}")

def preprocess_function(sample,padding="max_length"):
    # tokenize inputs
    model_inputs = tokenizer(sample["input"], max_length=max_source_length, padding=padding, truncation=True)

    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(text=sample["labels"], max_length=max_target_length, padding=padding, truncation=True)
    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["input", "labels"])
print(f"Keys of tokenized dataset: {list(tokenized_dataset['train'].features)}")

# load model from the hub
if model_id.startswith("mistral"):
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map='auto')
elif 'llama' in model_id.lower():
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map='auto')
else:
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

metric = evaluate.load("rouge")

# helper function to postprocess text
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(sent_tokenize(label)) for label in labels]

    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {k: round(v * 100, 4) for k, v in result.items()}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    return result

label_pad_token_id = -100
# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=8
)

from huggingface_hub import HfFolder
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

# Hugging Face repository id
if args.large:
    repository_id = f"models_{args.task}/{args.dataset}/{args.language}/{args.domain}/setting_{args.setting}/flan_large_{args.fold}_{args.template_name}"
elif args.small:
    repository_id = f"models_{args.task}/{args.dataset}/{args.language}/{args.domain}/setting_{args.setting}/flan_small_{args.fold}_{args.template_name}"
else:
    if args.data_filter=="folds":
        repository_id = f"models_{args.task}/{args.dataset}/{args.language}/{args.domain}/setting_{args.setting}/flan_base_{args.fold}_{args.template_name}"
    else:
        repository_id = f"models_{args.task}/{args.dataset}/{args.language}/{args.domain}/random/flan_base_{args.fold}_{args.num_examples}_{args.template_name}"

# Define training args
training_args = Seq2SeqTrainingArguments(
    output_dir=repository_id,
    do_eval=args.evaluate,
    do_train=not args.evaluate,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1000,
    predict_with_generate=True,
    fp16=False, # Overflows with fp16
    learning_rate=5e-5,
    num_train_epochs=10,
    # logging & evaluation strategies
    logging_dir=f"{repository_id}/logs",
    logging_strategy="steps",
    logging_steps=50,
    save_strategy="epoch",
    save_total_limit=1,
    # metric_for_best_model="overall_f1",
    # push to hub parameters
    push_to_hub=False,
#    gradient_checkpointing=True,
#    gradient_accumulation_steps=4,
#    optim='adafactor',

)

# Create Trainer instance
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    compute_metrics=compute_metrics,
)

if not args.evaluate:
    trainer.train()
    trainer.evaluate()
else:
    with open(os.path.join("..", args.dataset, "english", "ontology.json")) as json_file:
        ontology = json.load(json_file)

    if args.task=="intents":
        intent_desc_dict = {key:ontology["intents"][key]["description"][14:-1] for key in ontology["intents"].keys() if "general" in ontology["intents"][key]["domain"] or args.domain in ontology["intents"][key]["domain"]}
    elif args.task=="slots":
        intent_desc_dict = {key:ontology["slots"][key]["description"] for key in ontology["slots"].keys() if "general" in ontology["slots"][key]["domain"] or args.domain in ontology["slots"][key]["domain"]}
    intents_or_slots_list = sorted(list(intent_desc_dict.keys()))
    num_intents = len(intent_desc_dict)
    
    output = trainer.predict(tokenized_dataset["test"], max_new_tokens=max_target_length)
    outputs = tokenizer.batch_decode(output.predictions, skip_special_tokens=True)
    
    labels = [[idx for idx in label if idx!=-100] for label in tokenized_dataset["test"]["labels"]]
    labels_decoded = tokenizer.batch_decode(labels, skip_special_tokens=True)
    #mod_name = "_".join(args.model_name.split("/")[3:-1])
    mod_name = "_".join(args.model_name.split("/"))

    outputs = outputs_pipeline
    assert len(labels_decoded)==len(outputs)

    if args.task=="intents":    
        outputs = [outputs[i*num_intents:(i*num_intents+num_intents)] for i in range(int(len(outputs) / num_intents + 1))][:-1]
        outputs = [[1 if "yes" in int_out else 0 for int_out in output] for output in outputs]
    if args.task=="slots":
        if not args.template_name == "xtremeuplike":
            outputs = [outputs[i*num_intents:(i*num_intents+num_intents)] for i in range(int(len(outputs) / num_intents + 1))][:-1]
            outputs = [{slot:pred_value for slot, pred_value in zip(intents_or_slots_list, sent_output) if pred_value!="unanswerable"} for sent_output in outputs]

    labels = [[idx for idx in label if idx!=-100] for label in tokenized_dataset["test"]["labels"]]
    #labels = tokenizer.batch_decode(tokenized_dataset["test"]["labels"], skip_special_tokens=True)
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    if args.task=="intents":
        labels = [labels[i*num_intents:(i*num_intents+num_intents)] for i in range(int(len(labels) / num_intents + 1))][:-1]
        labels = [[1 if "yes" in int_out else 0 for int_out in output] for output in labels]
        print("*******************")
        print("PERFORMANCE")
        print(f1_score(labels, outputs, average="micro"))
        if args.dataset=="hwu64":
            labels_ids = [row.index(1) for row in labels]
            outputs_ids = [row.index(1) if 1 in row else random.sample(list(range(len(row))), 1)[0] for row in outputs]
            print(f"Accuracy: "+str(accuracy_score(labels_ids, outputs_ids)))
    elif args.task=="slots":
        labels = [labels[i*num_intents:(i*num_intents+num_intents)] for i in range(int(len(labels) / num_intents + 1))][:-1]
        labels = [{slot:label_value for slot, label_value in zip(intents_or_slots_list, sent_label) if label_value!="unanswerable"} for sent_label in labels]
        #print(labels)
        assert len(outputs)==len(labels)
        slot_list = set()
        true_positives = DefaultDict(lambda: 0)
        num_predicted = DefaultDict(lambda: 0)
        num_to_recall = DefaultDict(lambda: 0)
        for output, label in zip(outputs, labels):
            for slot in output.keys():
                slot_list.add(slot)
                num_predicted[slot] += 1

            for slot in label.keys():
                slot_list.add(slot)
                gold_text = label[slot]
                num_to_recall[slot] += 1
                if slot in output.keys() and output[slot] == gold_text:
                    true_positives[slot]+=1

        slot_type_f1_scores = DefaultDict()
        slot_type_precision = []
        slot_type_recall = []
        

        for slot in slot_list:
            slot_tp, slot_predicted, slot_to_recall = true_positives[slot], num_predicted[slot], num_to_recall[slot]
            slot_precision = precision(slot_tp, slot_predicted)
            slot_recall = recall(slot_tp, slot_to_recall)

            slot_type_precision.append(slot_precision)
            slot_type_recall.append(slot_recall)
            slot_type_f1_scores[slot] = calculate_f1(slot_precision, slot_recall)
            #print(slot, slot_tp, slot_predicted, slot_to_recall, slot_precision, slot_recall)

        averaged_f1 = np.mean(list(slot_type_f1_scores.values()))
        averaged_precision = np.mean(slot_type_precision)
        averaged_recall = np.mean(slot_type_recall)

        overall_true_positives = sum(true_positives.values())
        overall_num_predicted = sum(num_predicted.values())
        overall_num_to_recall = sum(num_to_recall.values())

        overall_precision = precision(overall_true_positives, overall_num_predicted)
        overall_recall = recall(overall_true_positives, overall_num_to_recall)
        overall_f1 = calculate_f1(overall_precision, overall_recall)
        print(overall_precision, overall_recall, overall_f1)

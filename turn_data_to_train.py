from argparse import ArgumentParser
import os
import json
from transformers import T5Tokenizer, T5ForConditionalGeneration
import numpy as np
from sklearn.metrics import f1_score
from googletrans import Translator
import random
from constants import SLOT_LISTS
from build_template import make_template, make_template_long_context, make_template_slot, make_template_slot_xtremeup, make_template_joint

parser = ArgumentParser(description='Arguments for training')
parser.add_argument('--dataset', type=str, help='Dataset name', default="multi3nlu")
parser.add_argument('--domain', type=str, help='Domain', default="banking")
parser.add_argument('--in_language', action='store_true', help='Whether the templates are in the target language')
parser.add_argument('--setting', type=int, help='Training data setting; [Options: 20, 10, 1]', default=10)
parser.add_argument('--fold', type=int, help='Fold', default=0)
parser.add_argument('--train', action='store_true', help='Whether it is training data')
parser.add_argument('--language', type=str, help='Language', default="english")
parser.add_argument('--template_name', type=str, help='Template key', default="none_none_none")
parser.add_argument('--task', type=str, help='Task working on; Options: [intents, slots]', default="intents")
parser.add_argument('--data_filter', type=str, help='How to filter the data: by folds/random', default="folds")
parser.add_argument('--num_examples', type=int, help='Number of random examples', default=500)

args = parser.parse_args()

data_dir = os.path.join(args.dataset, args.language, args.domain)

with open("templates.json") as json_file:
    templates_dict = json.load(json_file)
if args.task in ["intents", "slots"]:
  templates_dict = templates_dict[args.task]

if args.in_language:
  templates_dict = {intent: template for intent, template in templates_dict.items() if args.language in intent}

def get_data_by_fold(args):
  if args.dataset=="multi3nlu":
    total_folds = 20
  else:
    total_folds = 10

  if args.setting==10:
    fold = args.fold*2
    if args.train:
      folds = [fold, fold+1]
    else:
      folds = [i for i in range(total_folds) if not i in [fold, fold+1]]
  elif args.setting==20:
    fold = args.fold
    if args.train:
      folds = [fold]
    else:
      folds = [i for i in range(total_folds) if not i in [fold]]
  elif args.setting==1:
    fold = args.fold*2
    if args.train:
      folds = [i for i in range(total_folds) if not i in [fold, fold+1]]
    else:
      folds = [fold, fold+1]

  print(folds)
  data = []
  for fold_i in folds:
    with open(os.path.join(data_dir, f"fold{fold_i}.json")) as json_file:
      data += json.load(json_file)
  return data

def get_data_random(args):
  if args.dataset=="multi3nlu":
    total_folds = 20
  else:
    total_folds = 10
  data = []
  for fold_i in range(total_folds):
    with open(os.path.join(data_dir, f"fold{fold_i}.json")) as json_file:
      data += json.load(json_file)

  with open(os.path.join(data_dir, f"test_samples.txt"), "r") as tst_idx_file:
    test_indices = tst_idx_file.readlines()
  test_indices = [int(idx) for idx in test_indices]
  if not args.train:
    data_filtered = [data[idx] for idx in test_indices]
  else:
    indices_filtered = [idx for idx in range(len(data)) if idx not in test_indices]
    random.seed(args.fold)
    indices_filtered = random.sample(indices_filtered, args.num_examples)
    data_filtered = [data[idx] for idx in indices_filtered]

  return data_filtered


slot_desc_dict = None

if args.in_language:
  template = templates_dict[args.template_name+"_"+args.language]
else:
  if args.task in ["intents", "slots"]:
    template = templates_dict[args.template_name]
  else:
    template = {"intents":templates_dict["intents"][args.template_name], "slots":templates_dict["slots"][args.template_name]}
with open(os.path.join(args.dataset, "english", "ontology.json")) as json_file:
  ontology = json.load(json_file)
if args.task=="intents":
  intent_desc_dict = {key:ontology["intents"][key]["description"][14:-1] for key in ontology["intents"].keys() if "general" in ontology["intents"][key]["domain"] or args.domain in ontology["intents"][key]["domain"]}
  for intent, description in intent_desc_dict.items():
    if not description.startswith("to "):
      intent_desc_dict[intent] = description.replace("asking", "to ask")
  if args.template_name=="context_question":
      intent_desc_dict = {intent: "is the intent "+desc for intent, desc in intent_desc_dict.items()}
  if args.in_language:
    translator = Translator()
    intent_desc_dict = {intent:translator.translate(description, dest=args.language).text for intent, description in intent_desc_dict.items()}
    print(intent_desc_dict)
elif args.task=="slots":
  intent_desc_dict = {key:ontology["slots"][key]["description"] for key in ontology["slots"].keys() if "general" in ontology["slots"][key]["domain"] or args.domain in ontology["slots"][key]["domain"]}
elif args.task=="joint":
  slot_desc_dict = {key:ontology["slots"][key]["description"] for key in ontology["slots"].keys() if "general" in ontology["slots"][key]["domain"] or args.domain in ontology["slots"][key]["domain"]}
  intent_desc_dict = {key:ontology["intents"][key]["description"][14:-1] for key in ontology["intents"].keys() if "general" in ontology["intents"][key]["domain"] or args.domain in ontology["intents"][key]["domain"]}
  for intent, description in intent_desc_dict.items():
    if not description.startswith("to "):
      intent_desc_dict[intent] = description.replace("asking", "to ask")

intents_or_slots_list = sorted(list(intent_desc_dict.keys()))
if args.data_filter == "folds":
  data = get_data_by_fold(args)
elif args.data_filter == "random":
  data = get_data_random(args)

print(len(data))

all_labelled_data = []
for example in data:
  if args.task=="intents":
    if args.template_name.endswith("long_context"):
        labelled_data = make_template_long_context(example, intent_desc_dict, template, intents_or_slots_list, args)
    else:
        labelled_data = make_template(example, intent_desc_dict, template, intents_or_slots_list, args)
  elif  args.task=="slots":
    if not args.template_name=="xtremeuplike":
      labelled_data = make_template_slot(example, intent_desc_dict, template, intents_or_slots_list, args)
    else:
      labelled_data = make_template_slot_xtremeup(example, intent_desc_dict, template, intents_or_slots_list, args)
  elif  args.task=="joint":
    labelled_data = make_template_joint(example, intent_desc_dict, slot_desc_dict, template, intents_or_slots_list, args)
  if not labelled_data is None:
    all_labelled_data+=labelled_data

print(len(all_labelled_data))

if args.train:
  if args.in_language:
    file_name = f"train_{args.fold}_{args.template_name}_{args.setting}_in_{args.language}_{args.domain}_{args.task}.json"
  else:
    if args.data_filter == "folds":
      file_name = f"train_{args.fold}_{args.template_name}_{args.setting}_{args.domain}_{args.task}.json"
    else:
      file_name = f"train_random_{args.fold}_{args.num_examples}_{args.template_name}_{args.setting}_{args.domain}_{args.task}.json"
else:
  if args.in_language:
    file_name = f"test_{args.fold}_{args.template_name}_{args.setting}_in_{args.language}_{args.domain}_{args.task}.json"
  else:
    if args.data_filter == "folds":
      file_name = f"test_{args.fold}_{args.template_name}_{args.setting}_{args.domain}_{args.task}.json"
    else:
      file_name = f"test_random_{args.fold}_{args.num_examples}_{args.template_name}_{args.setting}_{args.domain}_{args.task}.json"

with open(os.path.join(args.dataset, args.language, file_name), 'w') as json_out:
  json.dump(all_labelled_data, json_out, ensure_ascii=True, indent=4)



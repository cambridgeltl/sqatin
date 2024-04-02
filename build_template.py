from constants import SLOT_LISTS

## make templates for intent classification
def make_template(example, intent_desc_dict, template, intents_or_slots_list, args):
  yes, no = "yes", "no"
  if args.in_language:
#    yes = translator.translate(yes, dest=args.language).text.lower()
#    no = translator.translate(no, dest=args.language).text.lower()
     yes, no = "yes", "no"
  sentence = example["text"]
  if str(sentence)=="nan":
    return None
  intents = example.get("intents")
  if intents is None:
  	intents = []
  data = []
  for intent in intents_or_slots_list: 
    data_dict = {}
    output = template.replace("SENTENCE", sentence)
    output = output.replace("QUESTION", intent_desc_dict[intent])
    data_dict["input"] = output
    if intent in intents:
      data_dict["labels"] = yes
    else:
      data_dict["labels"] = no
    data.append(data_dict)
  return data



def make_template_long_context(example, intent_desc_dict, template, intents_or_slots_list, args):
  yes, no = "yes", "no"
  if args.in_language:
#    yes = translator.translate(yes, dest=args.language).text.lower()
#    no = translator.translate(no, dest=args.language).text.lower()
     yes, no = "yes", "no"
  sentence = example["text"]
  if str(sentence)=="nan":
    return None
  intents = example.get("intents")
  if intents is None:
        intents = []
  data_dict = {}
  output = template.replace("SENTENCE", sentence)
  options = "\n".join([intent_desc_dict[intent] for intent in intent_desc_dict.keys()])
  output = output.replace("OPTIONS", options)
  labels = "$$".join([intent_desc_dict[intent_label] for intent_label in intents])
  data_dict["input"] = output
  data_dict["labels"] = labels
  
  return [data_dict]




def make_template_slot(example, intent_desc_dict, template, intents_or_slots_list, args):
  sentence = example["text"]
  if str(sentence)=="nan":
    return None
  slots = example.get("slots")

  if slots is None:
    slots = {}
  else:
    if isinstance(slots, dict):
      if slots is None:
        slots = {}
      else:
        slots = {slot: slots[slot]['text'] for slot in slots.keys()}
    elif isinstance(slots, list):
      if (slots is None) or (slots==[]):
        slots = {}
      else:
        slot_types = set([list(slot.keys())[0] for slot in slots])
        if len(slot_types)==len(slots):
          slots = {list(slot.keys())[0]:list(slot.values())[0]["text"] for slot in slots}
        else:
          slot_types_to_values = {}
          for slot_type in slot_types:
            slot_values = [slot[slot_type]["text"] for slot in slots if list(slot.keys())[0]==slot_type]
            if len(slot_values)>1:
              slot_types_to_values[slot_type] = ",".join(slot_values)
              slot_types_to_values[slot_type] = slot_values[0]
            else:
              slot_types_to_values[slot_type] = slot_values[0]
        
          slots = slot_types_to_values
  
  data = []
  for slot in intents_or_slots_list:
    data_dict = {}
    output = template.replace("SENTENCE", sentence)
    output = output.replace("QUESTION", intent_desc_dict[slot])
    data_dict["input"] = output
    if slot in list(slots.keys()):
      data_dict["labels"] = slots[slot]
    else:
      data_dict["labels"] = "unanswerable"
    data.append(data_dict)
  return data

def make_template_slot_xtremeup(example, intent_desc_dict, template, intents_or_slots_list, args):
  text = example["text"]
  
  slot_dict = example.get("slots")
  if slot_dict is None or slot_dict=={}:
    slots_values = ""
  else:
    slots_values = []
    for slot, value_dict in slot_dict.items():
      slots_values.append(" ".join([slot.upper(), value_dict["text"]]))
    slots_values = "$$".join(slots_values)

  slot_list = SLOT_LISTS[args.domain]
  inputs = template.replace("SLOT_LIST", slot_list).replace("SENTENCE", example["text"])
  return [{"input":inputs,"labels":slots_values}]


def make_template_joint(example, intent_desc_dict, slot_desc_dict, template, args):
  intents_list = sorted(intent_desc_dict.keys())
  slots_list = sorted(slot_desc_dict.keys())
  template_intent = template["intents"]
  template_slot = template["slots"]
  sentence = example["text"]
  slots = example.get("slots")
  intents = example.get("intents")
  if intents is None:
    intents = []
  if slots is None:
    slots = {}
  else:
    slots = {slot: slots[slot]['text'] for slot in slots.keys()}
  data = []
  
  for intent in intents_list: 
    data_dict = {}
    output = template_intent.replace("SENTENCE", sentence)
    output = output.replace("QUESTION", intent_desc_dict[intent])
    data_dict["input"] = output
    if intent in intents:
      data_dict["labels"] = "yes"
    else:
      data_dict["labels"] = "no"
    data.append(data_dict)
  for slot in slots_list:
    data_dict = {}
    output = template_slot.replace("SENTENCE", sentence)
    output = output.replace("QUESTION", slot_desc_dict[slot])
    data_dict["input"] = output
    if slot in list(slots.keys()):
      data_dict["labels"] = slots[slot]
    else:
      data_dict["labels"] = "unanswerable"
    data.append(data_dict)
  return data
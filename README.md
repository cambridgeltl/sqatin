## Code and data for the Paper "SQATIN: Supervised Instruction Tuning Meets Question Answering for Improved Dialogue NLU"

The repository contains code and data to replicate the experiments in the paper. 

Two main scripts are `turn_data_to_train.py` and `flan_finetune.py`.

```
turn_data_to_train.py
    Modifies the input data into instruction-based QA pairs:
        --dataset: which dataset to load [multi3nlu; clinc150]
        --domain: which domain to load [for multi3nlu -- banking/hotels; for clinc150 -- auto_and_commute/banking/credit_cards/home/kitcehn_and_dining/meta/small_talk/travel/utility/work]
        --setting: which data setiing to use: [20/10/0 for multi3nlu or 10 for clinc150]
        --fold: index of fold to load
        --train: whether to build test set/train set (by default -- test set)
        --template_name: which template to use [none_none_none/usersaid_QUESTION_none]
        --task: which task to us [intents/slots]
        --data_filter: How to filter the data: by folds/random
        --num_examples: Number of random examples if random data_filter
```

```
flan_finetune.py
    Trains and evaluates the model for SQATIN
    --dataset: which dataset to load [multi3nlu; clinc150]
    --fold: index of fold to load
    --template_name: which template to use [none_none_none/usersaid_QUESTION_none]
    --evaluate: if the model needs to be evaluated (rather than trained)
    --large/--small/--xlarge: model size of Flan to load
    --model_type: instr/qa for SQATIN/QA-FT
    --model_name: name of model (directory to load model from)
    --domain: which domain to load [for multi3nlu -- banking/hotels; for clinc150 -- auto_and_commute/banking/credit_cards/home/kitcehn_and_dining/meta/small_talk/travel/utility/work]
    --setting: which data setiing to use: [20/10/0 for multi3nlu or 10 for clinc150]
    --task: which task to us [intents/slots]
    --data_filter: How to filter the data: by folds/random
    --num_examples: Number of random examples if random data_filter
```

An example data loading+training script is available `train_slots.sh`.


#### Citation

The paper is to appear in NAACL-2024. While the proceedings of the conference are not openly available, please refer to the arxiv paper. 

```
@article{razumovskaia2023sqatin,
  title={SQATIN: Supervised Instruction Tuning Meets Question Answering for Improved Dialogue NLU},
  author={Razumovskaia, Evgeniia and Glava{\v{s}}, Goran and Korhonen, Anna and Vuli{\'c}, Ivan},
  journal={arXiv preprint arXiv:2311.09502},
  year={2023}
}
```
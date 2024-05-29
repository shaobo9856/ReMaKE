import os.path
import sys
sys.path.append('..')
import json
import random
from easyeditor import FTHyperParams, IKEHyperParams, KNHyperParams, MEMITHyperParams, ROMEHyperParams, MENDHyperParams, SERACHparams
from easyeditor import BaseEditor
import torch
from sentence_transformers import SentenceTransformer, util
from easyeditor.models.ike import encode_ike_facts
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--editing_method', required=True, type=str) #
    parser.add_argument('--hparams_dir', required=True, type=str) #
    parser.add_argument('--data_dir', required=True, type=str)
    parser.add_argument('--ds_size', default=None, type=int)
    parser.add_argument('--metrics_save_dir', default='./', type=str)
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument("--source_lang", type=str, default="zh") 
    parser.add_argument("--backbone", type=str, default="chinese_llama7b")
    parser.add_argument("--search", type=str, default="")
    parser.add_argument("--slang1", type=str, default="") #
    parser.add_argument("--slang2", type=str, default="") #


    args = parser.parse_args()

    if args.editing_method == 'FT':
        editing_hparams = FTHyperParams
    elif args.editing_method == 'IKE':
        editing_hparams = IKEHyperParams
    else:
        raise NotImplementedError
    



    hparams = editing_hparams.from_hparams(args.hparams_dir)
    editor = BaseEditor.from_hparams(hparams)
    sentence_model = SentenceTransformer(hparams.sentence_model_name).to('cuda')
    print("！！！！！！！！！！！model_name！！！！！！！！",hparams.model_name)
    print("！！！！！！！！！！！K！！！！！！！！",hparams.k)
    
    langs = ['en','cz','de','du','es','fr','pt','ru','th','tr','vi','zh','af']
    for i in range(len(langs)):
        lang1 = langs[i]
        for j in range(len(langs)):
            lang2 = langs[j]
            if 1: # if not os.path.exists(os.path.join(args.metrics_save_dir, f'{args.backbone}_{lang1}2{lang2}.json')):
                # 只打印 enzh pair
                if lang1 == args.slang1 and lang2 == args.slang2:
                    with open(os.path.join("./data/MCounterFact/",f"mcounterfact_test_{lang1}{lang2}.json"), "r", encoding="utf-8") as f:
                        test_data = json.load(f)
                else: continue
                if args.ds_size is not None:
                    test_data = random.sample(test_data, args.ds_size)

                prompts_truth = [test_data_[lang1]['src'] for test_data_ in test_data] # edit in english
                prompts_test = [test_data_[lang2]['src'] for test_data_ in test_data] # test in chinese

                target_truth = [edit_data_[lang1]['alt'] for edit_data_ in test_data] # edit in english
                target_test = [edit_data_[lang2]['alt'] for edit_data_ in test_data] # test in chinese

                rephrase_prompts = [edit_data_[lang2]['rephrase'] for edit_data_ in test_data]  # 测试Generalization
                locality_prompts = [edit_data_[lang2]['loc'].split('nq question: ')[-1] for edit_data_ in test_data]  # 测试Locality
                locality_ans = [edit_data_[lang2]['loc_ans'] for edit_data_ in test_data]  # 测试Locality
                portability_prompts = [edit_data_[lang2]['portability']['New Question'] for edit_data_ in test_data]  # 测试Portability
                portability_ans = [edit_data_[lang2]['portability']['New Answer'] for edit_data_ in test_data]  # 测试Portability



        
                edited_inputs = {
                    'edited_english': {
                        'prompt': prompts_truth,
                        'ground_truth': target_truth
                    },
                }
                cross_inputs = {
                    'cross': {
                        'prompt': prompts_test,
                        'ground_truth': target_test
                    },
                }
                generalization_inputs = {
                    'rephrase': {
                        'prompt': rephrase_prompts,
                        'ground_truth': target_test
                    },
                }
                locality_inputs = {
                    'neighborhood': {
                        'prompt': locality_prompts,
                        'ground_truth': locality_ans
                    },
                }
                portability_inputs = {
                    'one_hop': {
                        'prompt': portability_prompts,
                        'ground_truth': portability_ans
                    },
                }
        
                subject = [edit_data_[lang1]['subject'] for edit_data_ in test_data]

                train_ds = []

                with open(os.path.join("./data/MCounterFact/",f"mcounterfact_test_{lang1}{lang2}.json"), "r", encoding="utf-8") as f:
                    test_data = json.load(f)

                for test_data_ in test_data:
                    tt = dict()
                    tt["prompt"] = test_data_[lang1]['src']
                    tt["target_new"] = test_data_[lang1]['alt']
        
                    tt["prompt_mt"] = test_data_[lang2]['src']
                    tt["target_new_mt"] = test_data_[lang2]['alt']
        
                    train_ds.append(tt)
                    del tt
                    
                    
                if args.editing_method == 'IKE':
                    device = torch.device(f'cuda:{hparams.device}')
                    print("1111")
                    encode_ike_facts(sentence_model, train_ds, hparams,lang1)
                    print("2222")

                    metrics, edited_model, _ = editor.edit_ike(
                        edited_inputs=edited_inputs,
                        cross_inputs=cross_inputs,
                        generalization_inputs=generalization_inputs,
                        locality_inputs=locality_inputs,
                        portability_inputs=portability_inputs,
                        keep_original_weight=True,
                        lang1=lang1,
                        lang2=lang2,
                        search="ikeor",
                        subject=subject,
                        train_ds=train_ds
                    )
        
                print(os.path.join(args.metrics_save_dir, f'{args.backbone}_{lang1}2{lang2}.json'))
                if args.source_lang != "en":
                    # 确保目录存在
                    os.makedirs(args.metrics_save_dir, exist_ok=True) 
                    # 保存 metrics
                    json.dump(metrics, open(os.path.join(args.metrics_save_dir, f'{args.backbone}_{lang1}2{lang2}.json'), 'w'), ensure_ascii=False, indent=4)
                else:
                    raise NotImplementedError()     
            
            


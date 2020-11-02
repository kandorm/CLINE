import os
import random
from datasets import Dataset, concatenate_datasets

random.seed(12345)

if __name__ == "__main__":
    ori_dataset = Dataset.load_from_disk('disk/enwiki_bookcorpus-tiny-disk')
    rep_dataset = Dataset.load_from_disk('disk/enwiki_bookcorpus-tiny-wrep-disk')

    ori_num = ori_dataset.num_rows
    rep_num = rep_dataset.num_rows
    rep_list = random.sample(range(rep_num), ori_num)
    start_idx = 0
    def dataset_merge(examples):
        input_ids = examples['input_ids']
        input_ids = [ids.detach().numpy().tolist() for ids in input_ids]
        global start_idx
        end_idx = start_idx + len(input_ids)
        slc_list = rep_list[start_idx:end_idx]
        print(start_idx, end_idx)
        start_idx = end_idx

        original_sent = []
        synonym_sent = []
        antonym_sent = []
        synonym_antonym_sent = []
        replace_label = []
        for s in slc_list:
            t_d = rep_dataset[s]
            original_sent.append(t_d['original_sent'])
            synonym_sent.append(t_d['synonym_sent'])
            antonym_sent.append(t_d['antonym_sent'])
            synonym_antonym_sent.append(t_d['synonym_antonym_sent'])
            replace_label.append(t_d['replace_label'])

        return {'input_ids': input_ids,
                'original_sent': original_sent,
                'synonym_sent': synonym_sent,
                'antonym_sent': antonym_sent,
                'synonym_antonym_sent': synonym_antonym_sent,
                'replace_label':replace_label}

    dataset = ori_dataset.map(dataset_merge,
                              batched=True,
                              batch_size=5000,
                              writer_batch_size=5000,
                              remove_columns=ori_dataset.column_names,
                              load_from_cache_file=True,
                              cache_file_name="./cache/wrep-tiny-train.arrow",
                              num_proc=1)
    dataset.set_format(type=None, columns=['input_ids', 'original_sent', 'synonym_sent', 'antonym_sent', 'synonym_antonym_sent', 'replace_label'])
    dataset.save_to_disk("enwiki_bookcorpus-tiny-lec-disk")

# -*- coding: utf-8 -*-

import fasttext
import os
import jieba
import json
import argparse
from multiprocessing import Pool
from functools import partial


EXT = '.jsonl'
POS_LABEL =  'not_ad'
NEG_LABEL =  'ad'

def train(train_file, word2vec_file, model_file, w2v_dim):
    ### Train the FastText classifier using the pre-trained embeddings
    if word2vec_file:
        model = fasttext.train_supervised(input=train_file, dim=w2v_dim, label_prefix="__label__", epoch=25, lr=1.0, wordNgrams=3, pretrainedVectors=word2vec_file)
    else:
        model = fasttext.train_supervised(input=train_file, dim=w2v_dim, label_prefix="__label__", epoch=25, lr=1.0, wordNgrams=3)

    ### Save the trained model
    model.save_model(model_file)


def test(test_file, model_file):
    if os.path.exists(model_file):
        model = fasttext.load_model(model_file)
    else:
        raise
    print(model.test(test_file))


def predict(predict_file, model_file, stop_words):
    with open(predict_file) as f:
        text_list = [line.strip() for line in f]
    predicted_label, _ = predict_list(text_list, model_file, stop_words)
    res = list([f"{i} {label[0][9:]} {float(score[0]):.2f}" for i, (label, score) in enumerate(zip(predicted_label[0], predicted_label[-1]))])
    print(res)


def predict_list(text_list, model_file, stop_words):
    model = fasttext.load_model(model_file)
    seg_list = []
    for text in text_list:
        seg_text = jieba_cut(text, stop_words)
        seg_list.append(seg_text)
    predicted_label = model.predict(seg_list)
    return predicted_label, text_list


def jieba_cut(text, stop_words):
    seg_text = jieba.cut(text)
    seg_text_clean = [word.strip() for word in seg_text if word not in stop_words]
    text = " ".join(seg_text_clean)
    text.replace("\n", " ")
    return text


def read_stop_words(stop_word_path):
    stop_words = set()
    with open(stop_word_path) as f:
        for line in f:
            line = line.strip()
            stop_words.add(line)
    return stop_words


def write_jsonl(f1, f2, res, text_list):
    for label, score, text in zip(res[0], res[-1], text_list):
        label = label[0][9:]
        json_dict = {"text": text, "score": float(score[0]), "label": label}
        line = json.dumps(json_dict) + "\n"
        if label == "ad":
            f2.write(line)
        elif label == "not_ad":
            f1.write(line)


def data_loader(file_path, batch=1024):
    with open(file_path) as f:
        tmp = []
        for line in f:
            line = line.strip()
            json_dict = json.loads(line)
            tmp.append(json_dict)
            if len(tmp) >= batch:
                yield tmp
                tmp = []
        if tmp:
            yield tmp


def clean_single_file(file_path, model_file, stop_words, out_path, batch=1024):

    def predict_json(json_list, model, stop_words):
        seg_list = []
        for json_dict in json_list:
            text = json_dict['content']
            seg_text = jieba_cut(text, stop_words)
            seg_list.append(seg_text)
        predicted_label = model.predict(seg_list)
        return predicted_label
    
    model = fasttext.load_model(model_file)
    # 如果输出目录不存在，则创建
    out_dir = os.path.dirname(out_path)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    filename = os.path.basename(file_path)
    path_, ext = os.path.splitext(filename)
    label_0_out_path  = os.path.join(out_path, path_ + "_fastText_cleaned" + ext)
    label_0_out_finished = os.path.join(out_path, path_ + "_fastText_cleaned" + ext + '.finished')
    if os.path.isfile(label_0_out_finished):
        return
    with open(label_0_out_path, "w") as f1:
        loader = data_loader(file_path, batch)
        predict = partial(predict_json, model=model, stop_words=stop_words)
        for json_dicts in loader:
                res = predict(json_dicts)
                for label, score, json_dict in zip(res[0], res[-1], json_dicts):
                    label = label[0][9:]
                    if label == POS_LABEL:
                        text = json_dict['content']
                        meta = json_dict['warc_headers']
                        json_dict = {"content": text, "meta": meta}
                        line = json.dumps(json_dict, ensure_ascii=False) + "\n"
                        f1.write(line)
    
    with open(label_0_out_finished, "w") as f1:
        f1.write(f'{label_0_out_finished} finished!')


def clean_datasets(file_dir, model_file, stop_words, out_path, num_process, batch=1024):
    if os.path.isdir(file_dir):
        file_list = [os.path.join(file_dir, filename) for filename in os.listdir(file_dir) if filename.endswith(EXT)]
    elif os.path.isfile(file_dir) and file_dir.endswith(EXT):
        file_list = [file_dir]
    else:
        raise

    idx = 0
    if num_process <= 1:
        for file_path in file_list:
            clean_single_file(file_path, model_file, stop_words, out_path, batch)
            idx += 1
            print(f"finish {idx}/{len(file_list)} files")
    else:
        clean_file = partial(clean_single_file, model_file=model_file, stop_words=stop_words, out_path=out_path, batch=batch)
        with Pool(num_process) as pool:
            for _ in pool.imap_unordered(clean_file, file_list):
                idx += 1
                print(f"finish {idx}/{len(file_list)} files")


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-I', '--input_file', type=str, help='input file')
    parser.add_argument('--pretrain_w2v', default="./w2v/cc.zh.300.vec", type=str, help='pretain w2v to load when training, if dont need set None')
    parser.add_argument('--w2v_dim', default=200, type=int, help='w2v embedding dim, tencent pretrain w2v is 200, detail see in fasttext\w2v\readme.txt')
    parser.add_argument('--model_file', default="./output_models/fasttext.bin", type=str, help='output model path')
    parser.add_argument('--stop_word_path', default="./data/stop_words.txt", type=str, help='stop words to drop')
    parser.add_argument('--output_path', default="./clean_res.jsonl", type=str, help='result visualization path')
    parser.add_argument('-M', '--mode', type=str, default="train", help='train: train, test:test, predict:predict, clean:clean_datasets')
    parser.add_argument('--num_process', type=int, default=1, help='num of fasttext process')
    
    args = parser.parse_args()

    stop_word_path = args.stop_word_path
    input_file     = args.input_file
    word2vec_file  = args.pretrain_w2v
    model_file     = args.model_file
    out_path       = args.output_path
    mode           = args.mode
    w2v_dim        = args.w2v_dim
    num_process    = args.num_process

    stop_words     = read_stop_words(stop_word_path)

    # train the model
    if mode == "train":
        train(input_file, word2vec_file, model_file, w2v_dim)

    # test val data
    if mode == "test":
        test(input_file, model_file)

    # predict a file with text lines in it and print every line
    if mode == "predict":
        predict(input_file, model_file, stop_words)

    # clean files, input a data dir, and 
    if mode == "clean":
        clean_datasets(input_file, model_file, stop_words, out_path, num_process)

if __name__ == "__main__":
    main()

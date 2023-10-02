import os
import re
import random
import jieba


def count_chinese_characters(text):
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]')  # Pattern to match Chinese characters
    chinese_characters = re.findall(chinese_pattern, text)
    count = len(chinese_characters)
    return count


def segment_word(text):
    seg_text = jieba.cut(text.replace("\t"," ").replace("\n"," "))
    outline = " ".join(seg_text)
    return outline


def remove_stop_words(text, stop_words):
    word_list = text.split()
    word_list = [word for word in word_list if word not in stop_words]
    return word_list


def preprocess(file_path, data_list, label):   

    with open(file_path) as f:
        for line in f:
            line = line.strip()
            count = count_chinese_characters(line)
            if count < 20:
                continue
            outline = segment_word(line)
            outline = outline + "\t__label__" + label + "\n"
            data_list.append(outline)


def main():
    ad_list = []
    no_ad_list = []
    for label, file_path in file_dict.items():
        if label == "not_ad":
            data_list = no_ad_list
        elif label == "ad":
            data_list = ad_list
        preprocess(file_path, data_list, label)
    
    random.shuffle(ad_list)
    random.shuffle(no_ad_list)
    test_rate = 0.1

    ad_split_idx = int(len(ad_list)*0.1)
    not_ad_split_idx = int(len(no_ad_list)*0.1)
    train_list = ad_list[ad_split_idx:] + no_ad_list[not_ad_split_idx:]
    test_list = ad_list[:ad_split_idx] + no_ad_list[:not_ad_split_idx]

    random.shuffle(train_list)
    random.shuffle(no_ad_list)

    train_path = output_path + ".train"
    test_path =  output_path + ".test"
    with open(train_path, "w") as out_f:
        for line in train_list:
            out_f.write(line)
    
    with open(test_path, "w") as out_f:
        for line in test_list:
            out_f.write(line)

if __name__ == "__main__":
    file_dict = {'ad': '/your/path/', 'ad': '/your/path/', 'not_ad': '/your/path/', 'not_ad': '/your/path/'}
    output_path = "/comp_robot/xiongyuda/fastText/open_source_clean_data.txt"
    main()
### 说明
**本仓库简单利用fasttext为中文文本做文本分类，分类低质量和非低质量，低质量主要为色情，广告，赌博等。**


main函数为启动函数，参数包括：
   - --input_file，type=str，输入的训练文件，或其他模式的输入文件或目录
   - --pretrain_w2v，default="./w2v/cc.zh.300.vec", type=str, 训练时加载的预训练的w2v，如不需要则可设置为None
   - --w2v_dim，default=200, type=int, 训练的w2v的embd dim，如有预训练的w2v需要与之一致。建议采用腾讯的w2v，见./w2v/readme.txt
   - --model_file，default="./output_models/fasttext.bin", type=str, 训练模型的输出路径，或其他模式的模型加载路径
   - --stop_word_path，default="./data/stop_words.txt", type=str, 停用词设置
   - --output_path，default="./clean_res.jsonl", type=str，清洗结果保存位置
   - --mode type=str, default="train", 启动式，包括训练，验证，预测，清洗四种。预测只是简单可视化，不会保存结果，保存结果请用清洗模式。清洗模式会遍历目录内所有.jsonl文件并清洗
   - --num_process，type=int, default=1, fasettext预测进程数，代表启动N个fasttext分类器

如若需要修改输入文件读取方式请修改 main.py 中的 data_loader()

如若需要修改待清洗的文件后缀请修改 main.py 中的 EXT

如若训练的时候修改了 label name 请在修改 main.py 中的 POS_LABEL 和 NEG_LABEL


模型分类精度与训练数据强相关，如需提高性能建议先收集更高质量的数据集。baseline数据在 ./data/clean_data.txt.train 和 ./data/clean_data.txt.test

数据预处理可以参考clean_data.py

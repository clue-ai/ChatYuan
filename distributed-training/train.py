import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os, time
from loguru import logger

# Importing the T5 modules from huggingface/transformers
from torch import nn

from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch import cuda
from data import prepare_data
from torch_optimizer import Adafactor
from dialogdataset import DialogDataSet
import os
import horovod.torch as hvd

hvd.init()
logger.info("hvd.local_rank:{} ".format(hvd.local_rank()))
logger.info("hvd.rank:{} ".format(hvd.rank()))
logger.info("hvd.local_size:{} ".format(hvd.local_size()))
logger.info("hvd.size:{} ".format(hvd.size()))

torch.cuda.set_device(hvd.local_rank())
os.environ["CUDA_VISIBLE_DEVICES"] = str(hvd.local_rank())
device = 'cuda'

model_params = {
    "MODEL": "./ChatYuan-large-v1",  # model_type
    "TRAIN_BATCH_SIZE": 2,  # training batch size, 2
    "VALID_BATCH_SIZE": 4,  # validation batch size,8
    "TRAIN_EPOCHS": 3,  # number of training epochs
    "LEARNING_RATE": 3e-4,  # learning rate
    "MAX_SOURCE_TEXT_LENGTH": 512,  # max length of source text, 512
    "MAX_TARGET_TEXT_LENGTH": 512,  # max length of target text, 512
    "SEED": 42,  # set seed for reproducibility
    "ACCUMULATION_STEP": 32,  # accumulation_step
}


def train(epoch, tokenizer, model, device, loader, optimizer, accumulation_step):
    """
    用于训练的方法
    Function to be called for training with the parameters passed from main function

    """

    model.train()
    time1 = time.time()
    for _, data in enumerate(loader, 0):
        y = data["target_ids"].to(device, dtype=torch.long)
        y_ids = y[:, :-1].contiguous()  # target, from start to end(except end of token, <EOS>). e.g. "你好吗？"
        lm_labels = y[:, 1:].clone().detach()  # target, for second to end.e.g."好吗？<EOS>"
        lm_labels[y[:,
                  1:] == tokenizer.pad_token_id] = -100  # releted to pad_token and loss. for detail, check here: https://github.com/Shivanandroy/T5-Finetuning-PyTorch/issues/3
        ids = data["source_ids"].to(device, dtype=torch.long)  # input. e.g. "how are you?"
        mask = data["source_mask"].to(device, dtype=torch.long)

        outputs = model(
            input_ids=ids,
            attention_mask=mask,
            decoder_input_ids=y_ids,
            labels=lm_labels,
        )
        loss = outputs[0]
        loss = loss.mean()
        loss = loss / accumulation_step
        loss.backward()

        # training_logger.add_row(str(epoch), str(_), str(loss))
        # console.logger.info(training_logger)
        if (_ + 1) % accumulation_step == 0:
            optimizer.step()
            optimizer.zero_grad()
            if (_ + 1) % (accumulation_step * 3) == 0:
                time2 = time.time()
                logger.info(
                    "step: " + str(_) + " epoch:" + str(epoch) + "-loss:" + str(loss) + "; iter time spent:" + str(
                        float(time2 - time1)))
                time1 = time.time()


def validate(tokenizer, model, loader, max_length):
    """
    用于验证的方法：输入用于验证的数据，返回模型预测的结果和正确的标签
    Function to evaluate model for predictions

    """
    model.eval()
    predictions = []
    actuals = []
    source_list = []
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            y = data['target_ids'].to(device, dtype=torch.long)
            ids = data['source_ids'].to(device, dtype=torch.long)
            mask = data['source_mask'].to(device, dtype=torch.long)
            source_text = data["source_text"]

            generated_ids = model.generate(
                input_ids=ids,
                attention_mask=mask,
                max_length=max_length,
                num_beams=2,
                repetition_penalty=2.5,
                length_penalty=1.0,
                early_stopping=True
            )
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in
                     generated_ids]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in y]
            logger.info("source is: {} \npreds is: {} \ntarget is: {}".format(source_text, preds, target))
            if _ % 1000 == 0:
                logger.info(f'Completed {_}')
            source_list.extend(source_text)
            predictions.extend(preds)
            actuals.extend(target)
    return predictions, actuals, source_list


# 训练类：整合数据集类、训练方法、验证方法，加载数据进行训练并验证训练过程的效果
def T5Trainer():
    output_dir = "./outputs"
    """
    T5 trainer
    """
    logger.info("trainer begin")
    os.makedirs(output_dir, exist_ok=True)
    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(model_params["SEED"])  # pytorch random seed
    np.random.seed(model_params["SEED"])  # numpy random seed
    torch.backends.cudnn.deterministic = True

    # logging
    logger.info(f"""[Model]: Loading {model_params["MODEL"]}...\n""")
    logger.info("gpu number!: {}".format(torch.cuda.device_count()))

    # tokenzier for encoding the text
    tokenizer = T5Tokenizer.from_pretrained(model_params["MODEL"])

    # Defining the model. We are using ChatYuan model and
    # added a Language model layer on top for generation of prediction.
    # Further this model is sent to device (GPU/TPU) for using the hardware.
    model = T5ForConditionalGeneration.from_pretrained(model_params["MODEL"])
    model.to(device)
    # logging
    logger.info(f"[Data]: Reading data...\n")

    # Importing the raw dataset

    # display_df(dataframe.head(2))

    train_data_list, val_data_list = prepare_data()

    # Creating the Training and Validation dataset for further creation of Dataloader
    train_dataset = DialogDataSet(
        train_data_list,
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"],
        model_params["TRAIN_BATCH_SIZE"] * model_params["ACCUMULATION_STEP"] * torch.cuda.device_count()
        # trick，手动丢弃多余数据
    )

    logger.info("length of training dataset is: {}".format(len(train_dataset)))

    val_dataset = DialogDataSet(
        val_data_list,
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"]
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=hvd.size(),
                                                                    rank=hvd.rank(), shuffle=True)
    # Defining the parameters for creation of dataloaders
    train_params = {
        "batch_size": model_params["TRAIN_BATCH_SIZE"],
        "num_workers": 0,
    }

    val_params = {
        "batch_size": model_params["VALID_BATCH_SIZE"],
        "shuffle": False,
        "num_workers": 0,
    }

    # Creation of Dataloaders for testing and validation.
    # This will be used down for training and validation stage for the model.
    training_loader = DataLoader(train_dataset, batch_size=train_params["batch_size"], shuffle=False,
                                 sampler=train_sampler)
    val_loader = DataLoader(val_dataset, **val_params)

    # T5训练optimizer建议使用Adafactor，见论文原文。
    optimizer = Adafactor(
        params=model.parameters(), lr=model_params["LEARNING_RATE"] / hvd.size()
    )
    optimizer = hvd.DistributedOptimizer(optimizer, backward_passes_per_step=model_params["ACCUMULATION_STEP"])
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    # Training loop
    logger.info(f"[Initiating Fine Tuning]...\n")
    logger.info("the length of dataloader is: {}".format(len(training_loader)))

    for epoch in range(model_params["TRAIN_EPOCHS"]):
        # 1) train for one epoch
        train(epoch, tokenizer, model, device, training_loader, optimizer, model_params["ACCUMULATION_STEP"])

        # 2) save model for each epoch
        if hvd.rank() == 0:
            logger.info(f"[Saving Model]...\n")
            path = os.path.join(output_dir, "model_files" + "_epoch_{}".format(epoch))
            model.save_pretrained(path)
            tokenizer.save_pretrained(path)

            # 3) evaluating test dataset
            logger.info(f"[Initiating Validation]...\n")
            with torch.no_grad():  # add 2022.10.4
                if epoch != model_params["TRAIN_EPOCHS"] - 1:
                    continue
                predictions, actuals, source = validate(tokenizer, model, val_loader,
                                                        model_params["MAX_TARGET_TEXT_LENGTH"])
                predict_path = output_dir + "epoch_{}".format(epoch) + "_predictions.csv"
                final_df = pd.DataFrame({"source_text": source, "Generated Text": predictions, "Actual Text": actuals})
                final_df.to_csv(predict_path, index=False, sep="\t")

    logger.info(f"[Validation Completed.]\n")
    logger.info(
        f"""[Model] Model saved @ {os.path.join(output_dir, "model_files")}\n"""
    )
    logger.info(
        f"""[Validation] Generation on Validation data saved @ {os.path.join(output_dir, 'predictions.csv')}\n"""
    )
    logger.info(f"""[Logs] Logs saved @ {os.path.join(output_dir, 'logs.txt')}\n""")


if __name__ == "__main__":
    T5Trainer()

# from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
# import torch
# #https://www.datacamp.com/tutorial/mistral-7b-tutorial
# #pip install transformers==4.36.2
# #pip install trl==0.10.1
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_use_double_quant=True,
# )
#
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
# # model_name = "mistralai/Mistral-7B-v0.3"
#

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,HfArgumentParser,TrainingArguments,pipeline, logging
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
import os,torch, wandb
from datasets import load_dataset
from trl import SFTTrainer
#환경설정 ㅔㅛ39_118로 해야됨

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 인덱스 0과 1에 해당하는 GPU를 가시화
GPU_NUM = 0 # 원하는 GPU 번호 입력
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device) # change allocation of current GPU
print ('Current cuda device ', torch.cuda.current_device()) # check


# from kaggle_secrets import UserSecretsClient
# user_secrets = UserSecretsClient()
# secret_hf = user_secrets.get_secret("HUGGINGFACE_TOKEN")
# secret_wandb = user_secrets.get_secret("wandb")

wandb.login(key = "4aeea09a0b400b4511b18e2b929008d07fb3b229")
run = wandb.init(
    project='Fine tuning Qwen2.5-1.5B-Instruct',
    job_type="training",
    anonymous="allow"
)

base_model = model_name #"/kaggle/input/mistral/pytorch/7b-v0.1-hf/1"
dataset_name = "owkin/medical_knowledge_from_extracts" #mlabonne/guanaco-llama2-1k"
new_model = "mistral_7b_guanaco"

#Importing the dataset
dataset = load_dataset(dataset_name, split="train")
# print(dataset["text"][100])
def combine_qa(example):
    combined_text = example["Question"] + " " + example["Answer"]
    return {"combined_text": combined_text}

new_ds = dataset.map(combine_qa)
print(new_ds)
bnb_config = BitsAndBytesConfig(
    load_in_4bit= True,
    bnb_4bit_quant_type= "nf4",
    bnb_4bit_compute_dtype= torch.bfloat16,
    bnb_4bit_use_double_quant= False,
)
#auto로 하면 2080을 주로 씀. 0으로 해야 3090을 주로 씀
model_ = AutoModelForCausalLM.from_pretrained(
        base_model,
        # load_in_4bit=True,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map= "auto",
        trust_remote_code=True,
)
model_.config.use_cache = False # silence the warnings
model_.config.pretraining_tp = 1
model_.gradient_checkpointing_enable()

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.padding_side = 'right'
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True
# print(tokenizer.add_bos_token, tokenizer.add_eos_token)
print(tokenizer.add_eos_token)

model = prepare_model_for_kbit_training(model_)
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj"]
)
model = get_peft_model(model, peft_config)

training_arguments = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=25,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="wandb"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=new_ds, #dataset,
    peft_config=peft_config,
    max_seq_length= None,
    dataset_text_field="combined_text",
    tokenizer=tokenizer,
    args=training_arguments,
    packing= False,
)

trainer.train()

trainer.model.save_pretrained(new_model)
wandb.finish()
model.config.use_cache = True


model.eval()

prompt = """### 질문:
What is the recommended dosage of drug X?
문맥:
According to recent studies, the recommended dosage of drug X is 20mg per day.

### 답변:"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))


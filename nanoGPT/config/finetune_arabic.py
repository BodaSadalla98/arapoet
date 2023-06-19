import time

out_dir = 'out-arapoet'
eval_interval = 50
eval_iters = 10
wandb_log = False # feel free to turn on
wandb_project = 'shakespeare'
wandb_run_name = 'ft-' + str(time.time())

dataset = 'arabic'
# init_from = 'gpt2-large'# this is the largest GPT-2 model
init_from = 'resume'

# only save checkpoints if the validation loss improves
always_save_checkpoint = False



#train has 39,074,196 tokens
# val has 4,356,545 tokens
# the number of examples per iter:
# 4 batch_size * 32 grad_accum * 1024 tokens * 4 worldsize = 524,288 tokens/iter
# it will take 74 iterations for one epoch (one pass through the dataset)
batch_size = 4
gradient_accumulation_steps = 32
max_iters = 10000

# finetune at constant LR
learning_rate = 3e-5
decay_lr = False

dtype = 'float16'
# This is post for making a Ara-poet, a generative model that produces Arabic poems

In this project, we aim to apply the idea in Andrej's video [here](https://www.youtube.com/watch?v=kCc8FmEb1nY&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index).
But instead of generating Shakespear's plays, we will generate Arabic poems. We will use the transfomer architecture to generate the text.

## Baseline

As a first step, we need to create a baseline to compare the results. we will make a simple bigram model, and then compare its results to our transformer-based model

## Adding self attention

we start by adding a self attention layer

### Experiment:

total steps: 5000
batch size: 32
lr: 1e-3
block size: 8
head size: 32

train loss 2.5016, val loss 2.5338

## adding multi head attention

We can scale the attention to be done in parallel across different heads, just like we do with filters in CNNs.

### Experiment:

total steps: 5000
batch size: 32
lr: 1e-3
block size: 8
number of heads: 4
head size: 8

train loss 2.4182, val loss 2.4449

## Adding Feed forward layer

adding normal Linea layer followed by Relu

### Experiment:

Same configuration as before

train loss 2.3751, val loss 2.3793

## Adding attention blocks

- One attention block has:

  - Self Attention heads
  - Feed Forward Module

- when we run this, we get worse results, although that we increased the model capacity. This is due to the fact that the model became so deep and it got harder to optimize

### Experiments

train loss 2.3979, val loss 2.4063

## Adding resudial connections

Q: why do we need to add projection layer before adding the two paths? my guess is it's needed to align the dimensions of the two paths before adding them.

### Experiments

train loss 2.2219, val loss 2.1991

## Adding Layer norm (Pre norm formulation)

- this is a small deviation from the original transformers paper where we apply layers norm before the transformation
- so we apply it before the Self multihead attention and before the FFW layer, and lastly, after the transformer blocks before the last Linear layer

### Experiments

train loss 2.2129, val loss 2.2015

## Adding Dropout

adding dropout just before getting back to resdual connection

- we can see that dropout made the results worse, as we still have a small network, we need to scale it first, then we can make user of dropout

### Experiments

train loss 2.3565, val loss 2.3654

## Scalling the network

### My code

block_size = 256
batch_size = 64
eval_iterations = 200
eval_interval = 500
total_steps = 5000

num_heads = 6
head_size = 64
n_layers = 6
n_embd = num_heads \* head_size
dropout_p = 0.2
lr = 5e-4

#### Results

### Andrej code:

#### Results

batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

- train loss 1.1540, val loss 1.7917

# Resources

- video
- dataset
- repo
- HF space


# Experiments

## Using HF clm training scripts

### Starting from pre-trained  aragpt-large checkpoint
- Poor results
- Exxaple output: `.'كان يا ما كان في قديم من في في في من في من من في لا في في  في في عن في في او في في و في في لم في في خشية من في اح في في صغيرة في في ابت في في اص في في قد في`
### Starting from pre-trained  English gpt2-large checkpoint

## Using NanoGPT 

### Starting from pre-trained  Andrej English gpt2-large checkpoint
- Promising results

- Example output: `وجاء تفكر حولها فيا خير مشتاق راي فءادي لي بزينة في جنبيه مغمد عين الصحاصح مشكل فيها يهون كفه لواهي ضريحي الفا لا تفادي ولكن مستندي بجودك في النوم المتايق اذا ما شءت فالوهم افانيني اليوم يحمد وتري اليها اجابت بها واكرمها وانت مع الخلق تشابهين الي رحمة الف هو لا يغرد فان تسمحي بخطب فاتاهما يوم يسمح بفضله ويلقي من وشحه فتقدمين اليه بنفسين الي المحطات والمولد لي وقد كان يعدو حينا لا يكون بانفسين بالامس ما احتوته في غمض اله في شراين الي ان عطف اله في التيه اضحي والملاءك في شانها متكبر مطهر بغير احد`



# What did I learn:
- fixed a bug in HF clm training script, and make a pull request about it (LINK TO PR)[link]
- learned more about casual LMs and how to train them
- on the SW side, learned how experts write training scripts 
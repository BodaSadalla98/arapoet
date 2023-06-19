import torch
import torch.nn as nn
from torch.nn import functional as F

with open('tiny_data.txt', 'r') as f:
    data = f.read()


chars = sorted(list(set(data)))
vocab_size = len(chars)
print(vocab_size)
print(''.join(chars))

ctoi = {k:i for i, k in enumerate(chars)}
itoc = {k:v for v,k in ctoi.items()}



encode  = lambda x:[ctoi[c] for c in x]
decode = lambda x: ''.join([itoc[c] for c in x])




data_enc = torch.tensor(encode(data), dtype=torch.long)



#################### HYPER PARAMS ################
device = 'cuda' if torch.cuda.is_available() else 'cpu'
## Split data into: train, val
n = int(0.9 * len(data_enc))   
train_data = data_enc[:n]
val_data = data_enc[n:]
# Note that this training examples would be un-rolled into more examples
block_size = 8
batch_size  = 4
block_size = 8
x = train_data[:block_size]
y = train_data[1:block_size+1]
eval_iterations = 10
eval_interval = 1000
total_steps = 10000
# for t in range(block_size):
    # context = x[:t+1]
    # target = y[t]
    # print(f' when input is {context}, the target is {target}')


def get_batch(split):
    data = train_data if split == 'train' else val_data
    index = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i: i+block_size] for i in index])
    y = torch.stack([data[i+1: i+block_size+1] for i in index])

    print('-----------', index.shape, x.shape, y.shape)
    return x.to(device),y.to(device)
    
def gen(num_chars):
    st = torch.zeros((1,1), dtype=torch.long).to(device)
    return decode(m.generate(st,num_chars)[0].tolist())


@torch.no_grad()
def calc_losses():
    out = {}

    m.eval()
    for split in ['train','val']:
        loss_sum = 0.0
        for i in range(eval_iterations):

            xs, ys = get_batch(split=split)
            _ , loss = m(xs,ys)
            loss_sum += loss

        out[split] = loss_sum/eval_iterations
    
    m.train()
    return out
       

class BigramModel(nn.Module):
    def __init__(self, vocab_size) -> None:
        super().__init__()

        self.embed = nn.Embedding(vocab_size,vocab_size)
    def forward(self, idx, targets = None):
        loss = None
        logits = self.embed(idx)
        # print(logits.shape, targets.shape)
        
        if not targets == None:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    def generate(self, context, max_no):
        size = context.shape[1]

        context = context.to(device)
        for _ in range(max_no):
            cur = context[:, -size:]  # B, T
            # print('1',cur.shape)
            logits, _ = self(cur) # B*T , C 
            # print('2,',logits.shape)

            ## This works   for any context size, we average the logits for all characters so that we can do softmax on them
            # logits = logits.mean(1)

            ### THis is the method they used in the video, they only took the logits of the last character to predict the next one
            logits = logits[:,-1,:]


            
            # print('3,',logits.shape)
            props = F.softmax(logits, dim=1)# B*T , C
            # print('soft max',props.shape)
            # props = props.squeeze(0).view(-1, size) # B,T
            # print('probs 1',props.shape)
            next_char = torch.multinomial(props, 1)  # B,1
            # print(next_char.shape)
            context = torch.cat((context, next_char), dim =1) # B, T+1
        return context
    

m = BigramModel(vocab_size).to(device)
print(next(m.parameters()).device)



optimizer = torch.optim.AdamW(m.parameters(), lr=1e-4)

for step in range(total_steps):

    if step % eval_interval == 0:
        losses = calc_losses()
        print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb,yb = get_batch('train')

    logits, loss = m(xb,yb)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if not step % 1000:
        print(loss.item())

     



print(gen(100))
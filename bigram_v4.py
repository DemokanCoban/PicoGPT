import torch
import torch.nn as nn
from torch.nn import functional as F

#hyperparameters
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4 # self attention cannot handle large lr
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

torch.manual_seed(1337)

# wget
with open("data/input.txt", 'r', encoding="utf-8") as f:
    text = f.read()
    
# here are all the unique characters occurring in shakespeare
chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[ch] for ch in s] # takes string and returns a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # takes a list of int and returns a string of characters

# train and test split
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

#data loading
def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    """one head of self-attention"""
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        #tril is not a variable, and the pytorch way of doing it is registering it as a buffer to the module
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size))) 
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # B, T, head_size
        q = self.query(x) # B, T, head_size
        
        # compute attention scores, affinities
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, head_size) @ (B, , T) -> (B, T, T). the C**-0.5 is to normalize the variance, o.w. it will scale with the size of dim C and make softmax 'pikey' as opposed to diffused, important especially at the beginning of the training where we randomly initialize
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf")) #(B, T, T), masking so that info doesn't leak from the future. Its a decoder block
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        #weighted aggregation of the values
        v = self.value(x)
        out = wei @ v # (B, T, T) @ (B, T, head_size)
        return out
        
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out)) # we apply a linear transformation to the concatenated attention
        return out # we project back into the residual pathway
            
            
class FeedForward(nn.Module):
    """after communication b\w nodes happening, we allow nodes to process what they've just communicated via feeding those nodes one by one to this feedforward"""
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), # the multiplier 4 is to follow the same structure with the original papers suggestion; in their case, it was 512 to 2048
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), # in the similar vein with the block, we project to the residuals dimension with this one
            nn.Dropout(dropout), # right before residual pathway
        )
            
    def forward(self, x):
        return self.net(x)
    

class Block(nn.Module):
    """Transformer block: communication followed by computation"""
    
    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_nead: the no of heads we'd like to have
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
    def forward(self, x):
        #x = self.sa(x)
        x = x + self.sa(self.ln1(x)) # adding residuals and applying layernorm to x per token. Keep in mind that its different from the original attention paper, where we apply norm before, not after
        #x = self.ffwd(x)
        x = x + self.ffwd(self.ln2(x)) #adding residuals
        
        return x


# simple bigram model
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layernorm
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
        
    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B, T, C_embed)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        
        if targets is None:
            loss = None
            
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
            
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            # focus on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # softmax to get probs
            probs = F.softmax(logits, dim=-1) # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=-1) # (B, T+1)
        return idx
    
model = BigramLanguageModel()
m = model.to(device)
# no of parameters
print(sum(p.numel() for p in m.parameters())/1e6, "M parameters")

# creating the optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    # we evaluate loss every once in a while
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
    #sample a batch of data
    xb, yb = get_batch("train")
    
    #evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
#generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=550)[0].tolist()))
import torch
import torch.nn as nn

class PatchTSTClassifier(nn.Module):
    def __init__(self, seq_len=512, n_vars=2, patch_len=16, stride=16, 
                 n_classes=3, d_model=128, n_heads=4, n_layers=2, dropout=0.1, head_type="residual_bottleneck"):
        super().__init__()
        self.seq_len = seq_len
        self.n_vars = n_vars
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        
        # Patching
        # Number of patches: (L - P) / S + 1
        self.n_patches = (seq_len - patch_len) // stride + 1
        
        # Input Embedding: Patch -> d_model
        # Simple Linear projection of patch
        self.patch_embedding = nn.Linear(patch_len, d_model)
        
        # Positional Embedding (Learnable)
        self.position_embedding = nn.Parameter(torch.randn(1, self.n_vars, self.n_patches, d_model))
        
        # Transformer Encoder (Channel Independent handles vars as batch dim? Or just independent processing?)
        # "Channel Independent" usually means we treat (Batch * n_vars) as the effective batch size.
        # So we reshape (B, V, N, D) -> (B*V, N, D).
        
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=d_model*4, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        
        # Head
        self.flatten_dim = n_vars * self.n_patches * d_model
        
        self.head_type = head_type
        if head_type == "residual_bottleneck":
            # Residual Bottleneck: x + BN(x)
            # Bottleneck: Linear(Flat -> 256) -> ReLU -> Linear(256 -> Flat)
            self.bottleneck_down = nn.Linear(self.flatten_dim, 256)
            self.act = nn.ReLU()
            self.bottleneck_up = nn.Linear(256, self.flatten_dim)
            self.dropout = nn.Dropout(dropout)
            self.projection = nn.Linear(self.flatten_dim, n_classes)
        else:
            self.projection = nn.Linear(self.flatten_dim, n_classes)
            
    def forward(self, x):
        # x: (Batch, Vars, Length)
        B, V, L = x.shape
        
        # 1. Patching
        # Unfold: (B, V, N_Patches, Patch_Len)
        # We can use unfold on dimension 2.
        x_patched = x.unfold(dimension=2, size=self.patch_len, step=self.stride)
        # x_patched: (B, V, N, P)
        
        # 2. Embedding
        # (B, V, N, P) -> (B, V, N, D)
        x_emb = self.patch_embedding(x_patched)
        
        # 3. Add Positional Emb
        x_emb = x_emb + self.position_embedding[:, :, :self.n_patches, :]
        
        # 4. Channel Independence (Merge Batch and Vars)
        # (B, V, N, D) -> (B*V, N, D)
        x_ci = x_emb.view(B * V, self.n_patches, self.d_model)
        
        # 5. Transformer Encoder
        x_enc = self.encoder(x_ci)
        # (B*V, N, D)
        
        # 6. Reshape back
        x_out = x_enc.view(B, V, self.n_patches, self.d_model)
        
        # 7. Head (Channel Mixing)
        # Flatten: (B, V * N * D)
        x_flat = x_out.view(B, -1)
        
        if self.head_type == "residual_bottleneck":
            # Bottleneck path
            residual = x_flat
            
            x_bn = self.bottleneck_down(x_flat)
            x_bn = self.act(x_bn)
            x_bn = self.bottleneck_up(x_bn)
            
            # Residual Connection
            x_mixed = residual + self.dropout(x_bn)
            
            # Final Projection
            out = self.projection(x_mixed)
        else:
            out = self.projection(x_flat)
            
        return out # (B, n_classes) (Logits? Or Softmax? Usually Logits for CrossEntropyLoss)
        # Blueprint says "Softmax" at end. But CrossEntropyLoss in PyTorch expects Logits.
        # User prompt check: "Implement Focal Loss Only". Focal loss usually takes logits or probs depending on impl.
        # Standard PyTorch loss functions take logits. Let's return logits.
        # If model expects softmax, we add it. 
        # "Flatten -> Dense -> ... -> Softmax" in blueprint.
        # If I return logits, FocalLoss implementation must handle it.
        # I will return LOGITS to be safe for numerical stability. 
        # (Softmax is monotonic, so argmax is same).

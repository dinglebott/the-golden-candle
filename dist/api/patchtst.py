import torch
import torch.nn as nn


class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, mlp_ratio, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        hidden_dim = d_model * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        attn_input = self.norm1(x)
        attn_output, _ = self.attn(attn_input, attn_input, attn_input, need_weights=False)
        x = x + self.dropout1(attn_output)
        x = x + self.mlp(self.norm2(x))
        return x


class PatchTST(nn.Module):
    def __init__(self, num_core_features, corr_input_dim, config):
        super().__init__()
        self.patch_len = config["patch_len"]
        self.patch_stride = config["patch_stride"]
        self.d_model = config["d_model"]
        self.corr_input_dim = corr_input_dim

        num_patches = 1 + (config["lookback"] - self.patch_len) // self.patch_stride
        if num_patches <= 0:
            raise ValueError("lookback must be at least as large as patch_len")

        core_patch_dim = num_core_features * self.patch_len
        self.input_projection = nn.Linear(core_patch_dim, self.d_model)
        self.positional_embedding = nn.Parameter(torch.zeros(1, num_patches, self.d_model))
        self.input_dropout = nn.Dropout(config["dropout"])

        if corr_input_dim > 0:
            corr_patch_dim = corr_input_dim * self.patch_len
            self.corr_projection = nn.Linear(corr_patch_dim, self.d_model)
            self.corr_adapter = nn.Sequential(
                nn.LayerNorm(self.d_model),
                nn.Linear(self.d_model, self.d_model),
                nn.GELU(),
                nn.Dropout(config["dropout"]),
            )
        else:
            self.corr_projection = None
            self.corr_adapter = None

        self.shared_blocks = nn.ModuleList([
            EncoderBlock(self.d_model, config["num_heads"], config["mlp_ratio"], config["dropout"])
            for _ in range(config["base_encoder_blocks"])
        ])
        self.task_blocks = nn.ModuleList([
            EncoderBlock(self.d_model, config["num_heads"], config["mlp_ratio"], config["dropout"])
            for _ in range(config["branch_encoder_blocks"])
        ])

        head_hidden = max(self.d_model // 2, 32)
        self.head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, head_hidden),
            nn.GELU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(head_hidden, 1),
        )

        nn.init.trunc_normal_(self.positional_embedding, std=0.02)

    def patchify(self, x):
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.patch_stride)
        patches = patches.permute(0, 1, 3, 2).contiguous()
        return patches.view(x.size(0), patches.size(1), -1)

    def forward(self, core_inputs, corr_inputs=None):
        x = self.patchify(core_inputs)
        x = self.input_projection(x)
        x = self.input_dropout(x + self.positional_embedding[:, :x.size(1)])

        for block in self.shared_blocks:
            x = block(x)

        corr_context = 0.0
        if (
            self.corr_projection is not None
            and corr_inputs is not None
            and corr_inputs.size(-1) > 0
        ):
            corr_context = self.patchify(corr_inputs)
            corr_context = self.corr_projection(corr_context)
            corr_context = self.corr_adapter(corr_context)

        x = x + corr_context
        for block in self.task_blocks:
            x = block(x)

        return self.head(x.mean(dim=1)).squeeze(-1)

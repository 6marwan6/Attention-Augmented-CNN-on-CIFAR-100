import torch
import torch.nn as nn
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class AugmentedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dk, dv, Nh, relative):
        super(AugmentedConv, self).__init__()
        
        self.in_channels = in_channels  # number of input channels
        self.out_channels = out_channels  # number of output channels
        self.kernel_size = kernel_size  # Size of the convolution kernel
        self.dk = dk  # Dimension of the query/key
        self.dv = dv  # Dimension of the value
        self.Nh = Nh  # number of attention heads
        self.relative = relative  # Whether to use relative positional encoding

        # Standard convolutional 
        # Produces output with reduced channels (out_channels - dv)
        self.conv_out = nn.Conv2d(self.in_channels, self.out_channels - self.dv, self.kernel_size, padding=1)

        # Convolutional layer to compute query, key, and value tensors
        # Output channels = 2 * dk (query + key) + dv (value)
        self.qkv_conv = nn.Conv2d(self.in_channels, 2 * self.dk + self.dv, kernel_size=1)

        # Convolutional layer to process the attention output
        self.attn_out = nn.Conv2d(self.dv, self.dv, 1)

    def forward(self, x):
        # Extract the batch size, height, and width of the input tensor
        batch, _, height, width = x.size()

        # Apply the standard convolutional layer
        conv_out = self.conv_out(x)

        # Compute the flattened query, key, and value tensors
        flat_q, flat_k, flat_v, q, k, v = self.compute_flat_qkv(x, self.dk, self.dv, self.Nh)

        # Compute attention logits by performing matrix multiplication between query and key
        logits = torch.matmul(flat_q.transpose(2, 3), flat_k)

        # Add relative positional encoding if enabled
        if self.relative:
            h_rel_logits, w_rel_logits = self.relative_logits(q)
            logits += h_rel_logits
            logits += w_rel_logits

        # Apply softmax to compute attention weights
        weights = F.softmax(logits, dim=-1)

        # Compute the attention output by multiplying weights with the value tensor
        attn_out = torch.matmul(weights, flat_v.transpose(2, 3))
        attn_out = torch.reshape(attn_out, (batch, self.Nh, self.dv // self.Nh, height, width))

        # Combine attention heads and process with the attention output layer
        attn_out = self.combine_heads_2d(attn_out)
        attn_out = self.attn_out(attn_out)

        # Concatenate the convolutional output and attention output along the channel dimension
        return torch.cat((conv_out, attn_out), dim=1)

    def compute_flat_qkv(self, x, dk, dv, Nh):
        # Compute query, key, and value tensors using the qkv_conv layer
        N, _, H, W = x.size()
        qkv = self.qkv_conv(x)

        # Split the qkv tensor into separate query, key, and value tensors
        q, k, v = torch.split(qkv, [dk, dk, dv], dim=1)

        # Split the query, key, and value tensors into multiple heads
        q = self.split_heads_2d(q, Nh)
        k = self.split_heads_2d(k, Nh)
        v = self.split_heads_2d(v, Nh)

        # Scale the query tensor by the square root of the head dimension
        dkh = dk // Nh
        q = q * (dkh ** -0.5)

        # Flatten the query, key, and value tensors for attention computation
        flat_q = torch.reshape(q, (N, Nh, dk // Nh, H * W))
        flat_k = torch.reshape(k, (N, Nh, dk // Nh, H * W))
        flat_v = torch.reshape(v, (N, Nh, dv // Nh, H * W))
        return flat_q, flat_k, flat_v, q, k, v

    def split_heads_2d(self, x, Nh):
        # Split the input tensor into multiple attention heads
        batch, channels, height, width = x.size()
        ret_shape = (batch, Nh, channels // Nh, height, width)
        split = torch.reshape(x, ret_shape)
        return split

    def combine_heads_2d(self, x):
        # Combine multiple attention heads back into a single tensor
        batch, Nh, dv, H, W = x.size()
        ret_shape = (batch, Nh * dv, H, W)
        return torch.reshape(x, ret_shape)

    def relative_logits(self, q):
        # Compute relative positional logits for height and width
        B, Nh, dk, H, W = q.size()

        # Transpose the query tensor for compatibility with relative positional encoding
        q = torch.transpose(q, 2, 4).transpose(2, 3)

        # Compute relative positional logits for width
        key_rel_w = nn.Parameter(torch.randn((2 * W - 1, dk), requires_grad=True)).to(device)
        rel_logits_w = self.relative_logits_1d(q, key_rel_w, H, W, Nh, "w")

        # Compute relative positional logits for height
        key_rel_h = nn.Parameter(torch.randn((2 * H - 1, dk), requires_grad=True)).to(device)
        rel_logits_h = self.relative_logits_1d(torch.transpose(q, 2, 3), key_rel_h, W, H, Nh, "h")

        return rel_logits_h, rel_logits_w

    def relative_logits_1d(self, q, rel_k, H, W, Nh, case):
        # Compute relative positional logits for a single dimension (height or width)
        rel_logits = torch.einsum('bhxyd,md->bhxym', q, rel_k)

        # Reshape and convert relative logits to absolute logits
        rel_logits = torch.reshape(rel_logits, (-1, Nh * H, W, 2 * W - 1))
        rel_logits = self.rel_to_abs(rel_logits)

        # Reshape and repeat logits for compatibility with attention computation
        rel_logits = torch.reshape(rel_logits, (-1, Nh, H, W, W))
        rel_logits = torch.unsqueeze(rel_logits, dim=3)
        rel_logits = rel_logits.repeat((1, 1, 1, H, 1, 1))

        # Adjust logits based on the dimension (height or width)
        if case == "w":
            rel_logits = torch.transpose(rel_logits, 3, 4)
        elif case == "h":
            rel_logits = torch.transpose(rel_logits, 2, 4).transpose(4, 5).transpose(3, 5)

        # Reshape logits for final attention computation
        rel_logits = torch.reshape(rel_logits, (-1, Nh, H * W, H * W))
        return rel_logits

    def rel_to_abs(self, x):
        # Convert relative positional logits to absolute positional logits
        B, Nh, L, _ = x.size()

        # Add padding to the last column
        col_pad = torch.zeros((B, Nh, L, 1)).to(device)
        x = torch.cat((x, col_pad), dim=3)

        # Flatten and pad the tensor
        flat_x = torch.reshape(x, (B, Nh, L * 2 * L))
        flat_pad = torch.zeros((B, Nh, L - 1)).to(device)
        flat_x_padded = torch.cat((flat_x, flat_pad), dim=2)

        # Reshape and slice to obtain absolute positional logits
        final_x = torch.reshape(flat_x_padded, (B, Nh, L + 1, 2 * L - 1))
        final_x = final_x[:, :, :L, L - 1:]
        return final_x

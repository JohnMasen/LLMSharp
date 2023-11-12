using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace LLAMA2Sharp
{
    [StructLayout(LayoutKind.Sequential)]
    public struct ModelWeights
    {
        
            // token embedding table
            public ReadOnlyMemory<float> token_embedding_table; // (vocab_size, dim)

            // weights for rmsnorms
            public ReadOnlyMemory<float> rms_att_weight; // (layer, dim) rmsnorm weights

            public ReadOnlyMemory<float> rms_ffn_weight; // (layer, dim)

            // weights for matmuls
            public ReadOnlyMemory<float> wq; // (layer, dim, dim)
            public ReadOnlyMemory<float> wk; // (layer, dim, dim)
            public ReadOnlyMemory<float> wv; // (layer, dim, dim)

            public ReadOnlyMemory<float> wo; // (layer, dim, dim)

            // weights for ffn
            public ReadOnlyMemory<float> w1; // (layer, hidden_dim, dim)
            public ReadOnlyMemory<float> w2; // (layer, dim, hidden_dim)

            public ReadOnlyMemory<float> w3; // (layer, hidden_dim, dim)

            // final rmsnorm
            public ReadOnlyMemory<float> rms_final_weight; // (dim,)

            // freq_cis for RoPE relatively positional embeddings
            public ReadOnlyMemory<float> freq_cis_real; // (seq_len, head_size/2)

            public ReadOnlyMemory<float> freq_cis_imag; // (seq_len, head_size/2)

            // (optional) classifier weights for the logits, on the last layer
            public ReadOnlyMemory<float> wcls;
    }
}

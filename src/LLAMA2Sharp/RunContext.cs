using System;
using System.Buffers;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace LLAMA2Sharp
{
    public class RunContext
    {
        // current wave of activations
        public Memory<float> x; // activation at current time stamp (dim,)
        public Memory<float> xb; // same, but inside a residual branch (dim,)
        public Memory<float> xb2; // an additional buffer just for convenience (dim,)
        public Memory<float> hb; // buffer for hidden dimension in the ffn (hidden_dim,)
        public Memory<float> hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
        public Memory<float> q; // query (dim,)
        public Memory<float> k; // key (dim,)
        public Memory<float> v; // value (dim,)
        public Memory<float> att; // buffer for scores/attention values (n_heads, seq_len)
        public Memory<float> logits; // output logits

        public Memory<ProbIndex> probindex; // buffer used in top-p sampling

        // kv cache
        public Memory<float> key_cache; // (layer, seq_len, dim)
        public Memory<float> value_cache; // (layer, seq_len, dim)
        public static RunContext FromModelHeader(ModelHeaderInfo info)
        {
            Memory<float> createMemory(int capacity)
            {
                return new Memory<float>(new float[capacity]);
            }
            return new RunContext
            {
                x = createMemory(info.Dims),
                xb = createMemory(info.Dims),
                xb2 = createMemory(info.Dims),
                hb = createMemory(info.Hidden_Dims),
                hb2 = createMemory(info.Hidden_Dims),
                q = createMemory(info.Dims),
                k = createMemory(info.Dims),
                v = createMemory(info.Dims),
                att = createMemory(info.N_heads * info.Seq_length),
                logits = createMemory(info.Vocab_Size),
                probindex = new Memory<ProbIndex>(new ProbIndex[info.Vocab_Size]),
                key_cache = createMemory(info.N_layers * info.Seq_length * info.Dims),
                value_cache = createMemory(info.N_layers * info.Seq_length * info.Dims)
            };
        }
    }

    public struct ProbIndex
    {
        public float Prob;
        public int Index;
    }
}

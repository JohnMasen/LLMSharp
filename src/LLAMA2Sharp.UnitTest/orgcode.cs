using System;
using System.Collections.Generic;
using System.IO.MemoryMappedFiles;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace LLAMA2Sharp.UnitTest
{
    [StructLayout(LayoutKind.Sequential)]
    public struct Config
    {
        public int dim; // transformer dimension
        public int hidden_dim; // for ffn layers
        public int n_layers; // number of layers
        public int n_heads; // number of query heads
        public int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
        public int vocab_size; // vocabulary size, usually 256 (byte-level)
        public int seq_len; // max sequence length
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct TransformerWeights
    {
        // token embedding table
        public float[] token_embedding_table; // (vocab_size, dim)

        // weights for rmsnorms
        public ArraySegment<float> rms_att_weight; // (layer, dim) rmsnorm weights

        public ArraySegment<float> rms_ffn_weight; // (layer, dim)

        // weights for matmuls
        public ArraySegment<float> wq; // (layer, dim, dim)
        public ArraySegment<float> wk; // (layer, dim, dim)
        public ArraySegment<float> wv; // (layer, dim, dim)

        public ArraySegment<float> wo; // (layer, dim, dim)

        // weights for ffn
        public ArraySegment<float> w1; // (layer, hidden_dim, dim)
        public ArraySegment<float> w2; // (layer, dim, hidden_dim)

        public ArraySegment<float> w3; // (layer, hidden_dim, dim)

        // final rmsnorm
        public float[] rms_final_weight; // (dim,)

        // freq_cis for RoPE relatively positional embeddings
        public float[] freq_cis_real; // (seq_len, head_size/2)

        public float[] freq_cis_imag; // (seq_len, head_size/2)

        // (optional) classifier weights for the logits, on the last layer
        public float[] wcls;
    }
    internal class orgcode
    {
        public (Config config, TransformerWeights weights) LoadWeights(string checkpoint)
        {
            Config config;
            TransformerWeights weights;
            try
            {
                using FileStream fileStream = new FileStream(checkpoint, FileMode.Open, FileAccess.Read);
                // Read in the config header
                byte[] configBytes = new byte[Marshal.SizeOf(typeof(Config))];
                if (fileStream.Read(configBytes, 0, configBytes.Length) != configBytes.Length) Environment.Exit(1);

                GCHandle handle = GCHandle.Alloc(configBytes, GCHandleType.Pinned);
                try
                {
                    IntPtr pointer = handle.AddrOfPinnedObject();
                    config = (Config)Marshal.PtrToStructure(pointer, typeof(Config))!;
                }
                finally
                {
                    handle.Free();
                }

                // Negative vocab size is a hacky way of signaling unshared weights. Bit yikes.
                bool sharedWeights = config.vocab_size > 0;
                config.vocab_size = Math.Abs(config.vocab_size);

                // Figure out the file size
                var fileSize = fileStream.Length; // size of the checkpoint file in bytes

                using var memoryMappedFile = MemoryMappedFile.CreateFromFile(fileStream, null, fileSize,
                    MemoryMappedFileAccess.Read, HandleInheritability.None, false);
                long configSizeInBytes = Marshal.SizeOf(typeof(Config));
                using var accessor = memoryMappedFile.CreateViewAccessor(configSizeInBytes,
                    fileSize - configSizeInBytes, MemoryMappedFileAccess.Read);
                weights = new TransformerWeights();

                CheckpointInitWeights(ref weights, ref config, accessor, sharedWeights);
            }
            catch (Exception)
            {
                throw;
            }
            return (config, weights);
        }
        private static void CheckpointInitWeights(ref TransformerWeights w, ref Config p, MemoryMappedViewAccessor accessor,
    bool sharedWeights)
        {
            long offset = 0;

            w.token_embedding_table = ReadFloatArray(accessor, ref offset, p.vocab_size * p.dim);
            w.rms_att_weight = ReadFloatArray(accessor, ref offset, p.n_layers * p.dim);
            w.wq = ReadFloatArray(accessor, ref offset, p.n_layers * p.dim * p.dim);
            w.wk = ReadFloatArray(accessor, ref offset, p.n_layers * p.dim * p.dim);
            w.wv = ReadFloatArray(accessor, ref offset, p.n_layers * p.dim * p.dim);
            w.wo = ReadFloatArray(accessor, ref offset, p.n_layers * p.dim * p.dim);
            w.rms_ffn_weight = ReadFloatArray(accessor, ref offset, p.n_layers * p.dim);
            w.w1 = ReadFloatArray(accessor, ref offset, p.n_layers * p.dim * p.hidden_dim);
            w.w2 = ReadFloatArray(accessor, ref offset, p.n_layers * p.hidden_dim * p.dim);
            w.w3 = ReadFloatArray(accessor, ref offset, p.n_layers * p.dim * p.hidden_dim);
            w.rms_final_weight = ReadFloatArray(accessor, ref offset, p.dim);
            int headSize = p.dim / p.n_heads;
            w.freq_cis_real = ReadFloatArray(accessor, ref offset, p.seq_len * headSize / 2);
            w.freq_cis_imag = ReadFloatArray(accessor, ref offset, p.seq_len * headSize / 2);

            if (sharedWeights) w.wcls = w.token_embedding_table;
        }
        private static float[] ReadFloatArray(MemoryMappedViewAccessor accessor, ref long offset, int size)
        {
            float[] array = new float[size];
            accessor.ReadArray(offset, array, 0, size);
            offset += sizeof(float) * (long)size;
            return array;
        }

        public (string[] vocab, float[] vocabScores,int maxTokenLength) ReadVocab(string filepath,int vocab_size)
        {
            string[] vocab = new string[vocab_size];
            float[] vocabScores = new float[vocab_size];
            int maxTokenLength;

            using (FileStream fs = new FileStream(filepath, FileMode.Open,
                       FileAccess.Read))
            using (BinaryReader reader = new BinaryReader(fs))
            {
                try
                {
                    maxTokenLength = reader.ReadInt32();

                    for (int i = 0; i < vocab_size; i++)
                    {
                        vocabScores[i] = reader.ReadSingle();

                        int len = reader.ReadInt32();
                        Span<byte> buffer = stackalloc byte[len]; // stack allocate buffer, assumes len is small
                        _ = reader.Read(buffer);

                        vocab[i] = Encoding.UTF8.GetString(buffer);
                    }
                }
                catch (EndOfStreamException)
                {
                    Console.Error.WriteLine("failed read");
                    throw;
                }
            }
            return (vocab, vocabScores, maxTokenLength);
        }

        public  (int[] tokens,int nTokens) BpeEncode(string text, string[] vocab, float[] vocabScores, int vocabSize, int maxTokenLength)
        {
            int[] tokens = new int[text.Length];
            int nTokens ;
            int StrLookup(string str, string[] vocab, int vocabSize)
            {
                for (int i = 0; i < vocabSize; i++)
                    if (str == vocab[i])
                        return i;
                return -1;
            }

            StringBuilder strBuffer = new StringBuilder(maxTokenLength * 2 + 1); // *2 for concat, +1 for null terminator

            // first encode every individual byte in the input string
            nTokens = 0; // the number of tokens
            foreach (char c in text)
            {
                strBuffer.Clear();
                strBuffer.Append(c);
                int id = StrLookup(strBuffer.ToString(), vocab, vocabSize);
                if (id == -1)
                {
                    Console.Error.WriteLine("not good");
                    throw new Exception("Encoding error");
                }

                tokens[nTokens] = id;
                nTokens++;
            }

            // merge the best consecutive pair each iteration, according to the scores in vocab_scores
            while (true)
            {
                float bestScore = float.MinValue;
                int bestId = -1;
                int bestIdx = -1;

                for (int i = 0; i < nTokens - 1; i++)
                {
                    // check if we can merge the pair (tokens[i], tokens[i+1])
                    strBuffer.Clear();
                    strBuffer.Append(vocab[tokens[i]]);
                    strBuffer.Append(vocab[tokens[i + 1]]);

                    int id = StrLookup(strBuffer.ToString(), vocab, vocabSize);
                    if (id != -1 && vocabScores[id] > bestScore)
                    {
                        // this merge pair exists in vocab! record its score and position
                        bestScore = vocabScores[id];
                        bestId = id;
                        bestIdx = i;
                    }
                }

                if (bestIdx == -1) break; // we couldn't find any more pairs to merge, so we're done

                // merge the consecutive pair (bestIdx, bestIdx+1) into new token bestId
                tokens[bestIdx] = bestId;
                // delete token at position bestIdx+1, shift the entire sequence back 1
                for (int i = bestIdx + 1; i < nTokens - 1; i++) tokens[i] = tokens[i + 1];
                nTokens--; // token length decreased
            }
            return (tokens, nTokens);
        }

        public struct O_ProbIndex
        {
            public float Prob;
            public int Index;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct O_RunState
        {
            // current wave of activations
            public float[] x; // activation at current time stamp (dim,)
            public float[] xb; // same, but inside a residual branch (dim,)
            public float[] xb2; // an additional buffer just for convenience (dim,)
            public float[] hb; // buffer for hidden dimension in the ffn (hidden_dim,)
            public float[] hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
            public float[] q; // query (dim,)
            public float[] k; // key (dim,)
            public float[] v; // value (dim,)
            public float[] att; // buffer for scores/attention values (n_heads, seq_len)
            public float[] logits; // output logits

            public O_ProbIndex[] probindex; // buffer used in top-p sampling

            // kv cache
            public float[] key_cache; // (layer, seq_len, dim)
            public float[] value_cache; // (layer, seq_len, dim)
        }

        public void Rmsnorm(float[] o, float[] x, ArraySegment<float> weight, int size)
        {
            // calculate sum of squares
            float ss = 0.0f;
            for (int j = 0; j < size; j++) ss += x[j] * x[j];
            ss /= size;
            ss += 1e-5f;
            ss = 1.0f / MathF.Sqrt(ss);

            // normalize and scale
            for (int j = 0; j < size; j++) o[j] = weight[j] * (ss * x[j]);
        }

        public void Matmul(float[] xout, float[] x, ArraySegment<float> w, int n, int d)
        {
            // W (d,n) @ x (n,) . xout (d,)
            Parallel.For(0, d, i =>
            {
                float val = 0.0f;
                for (int j = 0; j < n; j++) val += w[i * n + j] * x[j];
                xout[i] = val;
            });
        }

        public int Argmax(float[] probabilities, int configVocabSize)
        {
            int maxI = 0;
            float maxP = probabilities[0];
            for (int i = 1; i < configVocabSize; i++)
                if (probabilities[i] > maxP)
                {
                    maxI = i;
                    maxP = probabilities[i];
                }

            return maxI;
        }
        public void Softmax(float[] x, int xOffset, int size)
        {
            // find max value (for numerical stability)
            float maxVal = x[0 + xOffset];
            for (int i = 1; i < size; i++)
                if (x[i + xOffset] > maxVal)
                    maxVal = x[i + xOffset];
            // exp and sum
            float sum = 0.0f;
            for (int i = 0; i < size; i++)
            {
                x[i + xOffset] = (float)Math.Exp(x[i + xOffset] - maxVal);
                sum += x[i + xOffset];
            }

            // normalize
            for (int i = 0; i < size; i++) x[i + xOffset] /= sum;
        }

        public void Accum(float[] a, float[] b, int size)
        {
            for (int i = 0; i < size; i++) a[i] += b[i];
        }

        public void Transformer(int token, int pos, Config config, RunState state, TransformerWeights w)
        {
            // a few convenience variables
            int dim = config.dim;
            int hiddenDim = config.hidden_dim;
            int headSize = dim / config.n_heads;

            // copy the token embedding into x
            Array.Copy(w.token_embedding_table, token * dim, state.x, 0, dim);


            // forward all the layers
            for (int l = 0; l < config.n_layers; l++)
            {
                // attention rmsnorm
                Rmsnorm(state.xb, state.x, w.rms_att_weight[(l * dim)..], dim);

                // qkv matmuls for this position
                Matmul(state.q, state.xb, w.wq[(l * dim * dim)..], dim, dim);   
                Matmul(state.k, state.xb, w.wk[(l * dim * dim)..], dim, dim);
                Matmul(state.v, state.xb, w.wv[(l * dim * dim)..], dim, dim);

                // RoPE relative positional encoding: complex-valued rotate q and k by freq_cis in each head
                for (int i = 0; i < dim; i += 2)
                {
                    float q0 = state.q[i];
                    float q1 = state.q[i + 1];
                    float k0 = state.k[i];
                    float k1 = state.k[i + 1];
                    float fcr = w.freq_cis_real[pos * headSize / 2 + i % headSize / 2];
                    float fci = w.freq_cis_imag[pos * headSize / 2 + i % headSize / 2];
                    state.q[i] = q0 * fcr - q1 * fci;
                    state.q[i + 1] = q0 * fci + q1 * fcr;
                    state.k[i] = k0 * fcr - k1 * fci;
                    state.k[i + 1] = k0 * fci + k1 * fcr;
                }

                // save key,value at this time step (pos) to our kv cache
                int loff = l * config.seq_len * dim; // kv cache layer offset for convenience
                Array.Copy(state.k, 0, state.key_cache, loff + pos * dim, dim);
                Array.Copy(state.v, 0, state.value_cache, loff + pos * dim, dim);

                // multihead attention. iterate over all heads
                Parallel.For(0, config.n_heads, h =>
                {
                    // get the query vector for this head
                    int qOffset = h * headSize;

                    // attention scores for this head
                    int attOffset = h * config.seq_len;

                    // iterate over all timesteps, including the current one
                    for (int t = 0; t <= pos; t++)
                    {
                        // get the key vector for this head and at this timestep
                        int keyCacheOffset = loff + t * dim + h * headSize;

                        // calculate the attention score as the dot product of q and k
                        float score = 0.0f;
                        for (int i = 0; i < headSize; i++)
                            score += state.q[i + qOffset] * state.key_cache[i + keyCacheOffset];

                        score /= (float)Math.Sqrt(headSize);

                        // save the score to the attention buffer
                        state.att[t + attOffset] = score;
                    }


                    // softmax the scores to get attention weights, from 0..pos inclusively
                    Softmax(state.att, attOffset, pos + 1);

                    // weighted sum of the values, store back into xb
                    int xbOffset = h * headSize;
                    for (int i = xbOffset; i < xbOffset + headSize; i++) state.xb[i] = 0f;

                    for (int t = 0; t <= pos; t++)
                    {
                        // get the value vector for this head and at this timestep
                        int vOffset = loff + t * dim + h * headSize;

                        // get the attention weight for this timestep
                        float a = state.att[t + attOffset];

                        // accumulate the weighted value into xb
                        for (int i = 0; i < headSize; i++)
                            state.xb[i + xbOffset] += a * state.value_cache[i + vOffset];
                    }
                });

                ;

                // final matmul to get the output of the attention
                Matmul(state.xb2, state.xb, w.wo[(l * dim * dim)..], dim, dim);

                // residual connection back into x
                Accum(state.x, state.xb2, dim);

                // ffn rmsnorm
                Rmsnorm(state.xb, state.x, w.rms_ffn_weight[(l * dim)..], dim);

                // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
                // first calculate self.w1(x) and self.w3(x)
                Matmul(state.hb, state.xb, w.w1[(l * dim * hiddenDim)..], dim, hiddenDim);
                Matmul(state.hb2, state.xb, w.w3[(l * dim * hiddenDim)..], dim, hiddenDim);

                // F.silu; silu(x)=x*σ(x),where σ(x) is the logistic sigmoid
                for (int i = 0; i < hiddenDim; i++)
                    state.hb[i] *= (1.0f / (1.0f + (float)Math.Exp(-state.hb[i])));

                // elementwise multiply with w3(x)
                for (int i = 0; i < hiddenDim; i++) state.hb[i] *= state.hb2[i];

                // final matmul to get the output of the ffn
                Matmul(state.xb, state.hb, w.w2[(l * dim * hiddenDim)..], hiddenDim, dim);

                // residual connection
                Accum(state.x, state.xb, dim);
            }

            // final rmsnorm
            Rmsnorm(state.x, state.x, w.rms_final_weight, dim);

            // classifier into logits
            Matmul(state.logits, state.x, w.wcls, config.dim, config.vocab_size);
        }
    }
}

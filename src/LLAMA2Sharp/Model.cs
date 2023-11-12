using System;
using System.Buffers;
using System.Collections.Generic;
using System.IO;
using System.Net;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace LLAMA2Sharp
{
    public class Model
    {
        public ModelHeaderInfo Header { get; private set; }
        private Memory<float> weightBuffer;
        private readonly int headerSize= Marshal.SizeOf(typeof(ModelHeaderInfo));
        public ModelWeights Weights;
        public Model()
        { }

        public void LoadWeights(string file)
        {
            
            using var fs = File.OpenRead(file);
            
            using BinaryReader reader = new BinaryReader(fs);
            reader.BaseStream.Position += headerSize;//skip header
            long bytesToRead = reader.BaseStream.Length - headerSize;
            weightBuffer = new Memory<float>(new float[bytesToRead / 4]);
            var tmp = MemoryMarshal.AsBytes(weightBuffer.Span);
            if (reader.Read(tmp)!=bytesToRead)
            {
                throw new InvalidOperationException("error reading file");
            }
            
            int pos = 0;
            Weights.token_embedding_table = sliceBuffer(weightBuffer, ref pos, Header.Vocab_Size * Header.Dims);
            Weights.rms_att_weight = sliceBuffer(weightBuffer, ref pos, Header.LayersDim);
            Weights.wq = sliceBuffer(weightBuffer, ref pos, Header.LayersDim2);
            Weights.wk=sliceBuffer(weightBuffer,ref pos, Header.LayersDim2);
            Weights.wv=sliceBuffer(weightBuffer,ref pos ,Header.LayersDim2);
            Weights.wo= sliceBuffer(weightBuffer, ref pos, Header.LayersDim2);
            Weights.rms_ffn_weight=sliceBuffer(weightBuffer,ref pos,Header.LayersDim);
            Weights.w1 = sliceBuffer(weightBuffer, ref pos, Header.LayersDimHiddenDim);
            Weights.w2 = sliceBuffer(weightBuffer, ref pos, Header.LayersDimHiddenDim);
            Weights.w3 = sliceBuffer(weightBuffer, ref pos, Header.LayersDimHiddenDim);
            Weights.rms_final_weight = sliceBuffer(weightBuffer, ref pos, Header.Dims);
            Weights.freq_cis_real = sliceBuffer(weightBuffer, ref pos, Header.Seq_length* Header.HeadSizeHalf);
            Weights.freq_cis_imag = sliceBuffer(weightBuffer, ref pos, Header.Seq_length * Header.HeadSizeHalf);
            if (Header.SharedWeights)
            {
                Weights.wcls = Weights.token_embedding_table;
            }
        }
        private Memory<float> sliceBuffer(Memory<float> buffer,ref int position, int length)
        {
            var result= buffer.Slice(position, length);
            position += length;
            return result;
        }


        
        public void LoadHeader(string file)
        {
            byte[] buffer;
            using var fs = File.OpenRead(file);
            using BinaryReader reader= new BinaryReader(fs);
            buffer=reader.ReadBytes(headerSize);
            Header=MemoryMarshal.Cast<byte,ModelHeaderInfo>(buffer)[0];
        }

        public void Transform(int token,int pos,RunContext context)
        {
            int dims=Header.Dims;
            int hiddenDims = Header.Hidden_Dims;
            int headSize = Header.HeadSize;
            int halfHeadSize = Header.HeadSizeHalf;
            // copy the token embedding into x
            Weights.token_embedding_table.Span.Slice(token * dims, dims).CopyTo(context.x.Span);
            
            // forward all the layers
            for (int i = 0; i < Header.N_layers; i++)
            {
                
                // attention rmsnorm
                MathHelper.RMSNorm(context.xb.Span, context.x.Span, Weights.rms_att_weight.Span.Slice(i * dims, dims), dims);

                // qkv matmuls for this position
                MathHelper.MatMul(context.xb.Span, dims, dims, Weights.wq.Span.Slice(i * dims * dims, dims*dims),context.q.Span);
                //MathHelper.DumpSpan(context.q.Span);
                MathHelper.MatMul(context.xb.Span, dims, dims, Weights.wk.Span.Slice(i * dims * dims, dims * dims), context.k.Span);
                MathHelper.MatMul(context.xb.Span, dims, dims, Weights.wv.Span.Slice(i * dims * dims, dims * dims), context.v.Span);
                
                // RoPE relative positional encoding: complex-valued rotate q and k by freq_cis in each head
                //TODO: reimplement with vector?
                for (int j = 0; j < dims; j += 2)
                {
                    float q0 = context.q.Span[j];
                    float q1 = context.q.Span[j + 1];
                    float k0 = context.k.Span[j];
                    float k1 = context.k.Span[j + 1];
                    float fcr = Weights.freq_cis_real.Span[pos * halfHeadSize + j % headSize/2];
                    float fci = Weights.freq_cis_imag.Span[pos * halfHeadSize + j % headSize/2];
                    context.q.Span[j] = q0 * fcr - q1 * fci;
                    context.q.Span[j + 1] = q0 * fci + q1 * fcr;
                    context.k.Span[j] = k0 * fcr - k1 * fci;
                    context.k.Span[j + 1] = k0 * fci + k1 * fcr;
                }
                // save key,value at this time step (pos) to our kv cache
                int ioff = i * Header.Seq_length * dims;
                context.k.Span.CopyTo(context.key_cache.Span.Slice(ioff + pos * dims , dims));
                context.v.Span.CopyTo(context.value_cache.Span.Slice(ioff + pos * dims, dims));



                // multihead attention. iterate over all heads
                Parallel.For(0, Header.N_heads, h =>
                //for (int h = 0; h < Header.N_heads; h++)
                {
                    // get the query vector for this head
                    int qOffset = h * headSize;

                    // attention scores for this head
                    int attOffset = h * Header.Seq_length;

                    // iterate over all timesteps, including the current one
                    for (int t = 0; t <= pos; t++)
                    {
                        // get the key vector for this head and at this timestep
                        int keyCacheOffset = ioff + t * dims + h * headSize;

                        // calculate the attention score as the dot product of q and k
                        float score = 0.0f;
                        ReadOnlySpan<float> qSpan = context.q.Span.Slice(qOffset, headSize);
                        ReadOnlySpan<float> keySpan = context.key_cache.Span.Slice(keyCacheOffset, headSize);
                        MathHelper.WithSpan(qSpan, keySpan, (q, k) =>
                        {
                            score += Vector.Dot(q, k);
                        },
                        (q, k) =>
                        {
                            score += q * k;
                        });
                        score /= MathF.Sqrt(headSize);

                        // save the score to the attention buffer
                        context.att.Span[t + attOffset] = score;
                    }
                        // softmax the scores to get attention weights, from 0..pos inclusively
                    MathHelper.SoftMax(context.att.Span.Slice(attOffset, pos + 1));
                    // weighted sum of the values, store back into xb
                    int xbOffset = h * headSize;
                    context.xb.Slice(xbOffset, headSize).Span.Fill(0f);

                    for (int t = 0; t <= pos; t++)
                    {
                        // get the value vector for this head and at this timestep
                        int vOffset = ioff + t * dims + h * headSize;

                        // get the attention weight for this timestep
                        float a = context.att.Span[t + attOffset];
                        var xb = context.xb.Span.Slice(xbOffset, headSize);
                        var vcache=context.value_cache.Span.Slice(vOffset, headSize);
                        MathHelper.WithSpanWB(xb, vcache, (x, v) =>
                        {
                            return x+v * a;
                        }, (x, v) =>
                        {
                            return x+v * a;
                        });
                        
                    }


                })
                ;
                

                // final matmul to get the output of the attention
                MathHelper.MatMul(context.xb.Span,dims,dims, Weights.wo.Span.Slice(i * dims*dims, dims*dims), context.xb2.Span);

                // residual connection back into x
                MathHelper.Accum(context.x.Span, context.xb2.Span);

                // ffn rmsnorm
                MathHelper.RMSNorm(context.xb.Span, context.x.Span, Weights.rms_ffn_weight.Span.Slice(i * dims, dims),dims);

                // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
                // first calculate self.w1(x) and self.w3(x)
                MathHelper.MatMul(context.xb.Span, dims, hiddenDims, Weights.w1.Span.Slice(i * dims * hiddenDims, dims*hiddenDims),context.hb.Span);
                MathHelper.MatMul(context.xb.Span, dims, hiddenDims, Weights.w3.Span.Slice(i * dims * hiddenDims, dims*hiddenDims), context.hb2.Span);


                // F.silu; silu(x)=x*σ(x),where σ(x) is the logistic sigmoid
                MathHelper.SILU(context.hb.Span);
                
                // elementwise multiply with w3(x)
                MathHelper.Multiply(context.hb.Span, context.hb2.Span);
                
                // final matmul to get the output of the ffn
                MathHelper.MatMul(context.hb.Span, hiddenDims, dims, Weights.w2.Span.Slice(i * dims * hiddenDims,hiddenDims*dims), context.xb.Span);
                
                // residual connection
                MathHelper.Accum(context.x.Span, context.xb.Span);
                
            }
            
            // final rmsnorm
            MathHelper.RMSNorm(context.x.Span,context.x.Span,Weights.rms_final_weight.Span,dims);
            //if (pos == 10)
            //{
            //    //TODO: dump value_cache for item compare
            //    MathHelper.DumpSpan(context.x.Span);
            //}

            // classifier into logits
            MathHelper.MatMul(context.x.Span, dims, Header.Vocab_Size,  Weights.wcls.Span, context.logits.Span);
        }

    }
}

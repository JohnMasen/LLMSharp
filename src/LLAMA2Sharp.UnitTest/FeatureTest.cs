using System;
using System.Diagnostics;
using System.Numerics.Tensors;
using System.Reflection;
using System.Security.Cryptography.X509Certificates;
using static LLAMA2Sharp.UnitTest.Consts;
namespace LLAMA2Sharp.UnitTest
{
    [TestClass]
    public class FeatureTest
    {
        
        [TestMethod]
        public void LoadHeader()
        {
            orgcode org = new orgcode();
            var r = org.LoadWeights(WEIGHT_FILE_PATH);
            var c = r.config;

            Model model = new Model();
            model.LoadHeader(WEIGHT_FILE_PATH);
            Assert.AreEqual(c.dim, model.Header.Dims);
            Assert.AreEqual(c.hidden_dim, model.Header.Hidden_Dims);
            Assert.AreEqual(c.n_layers, model.Header.N_layers);
            Assert.AreEqual(c.n_heads, model.Header.N_heads);
            Assert.AreEqual(c.n_kv_heads, model.Header.N_kv_heads);
            Assert.AreEqual(c.seq_len, model.Header.Seq_length);
            Assert.AreEqual(c.vocab_size, model.Header.Vocab_Size);
        }

        [TestMethod]
        public void LoadWeights()
        {
            orgcode org = new orgcode();
            var r = org.LoadWeights(WEIGHT_FILE_PATH);
            var c = r.config;

            Model model = new Model();
            model.LoadHeader(WEIGHT_FILE_PATH);
            model.LoadWeights(WEIGHT_FILE_PATH);

            Assert.IsTrue(model.Weights.token_embedding_table.Span.SequenceEqual(r.weights.token_embedding_table));
            Assert.IsTrue(model.Weights.rms_att_weight.Span.SequenceEqual(r.weights.rms_att_weight));
            Assert.IsTrue(model.Weights.rms_ffn_weight.Span.SequenceEqual(r.weights.rms_ffn_weight));
            Assert.IsTrue(model.Weights.wq.Span.SequenceEqual(r.weights.wq));
            Assert.IsTrue(model.Weights.wk.Span.SequenceEqual(r.weights.wk));
            Assert.IsTrue(model.Weights.wv.Span.SequenceEqual(r.weights.wv));
            Assert.IsTrue(model.Weights.wo.Span.SequenceEqual(r.weights.wo));
            Assert.IsTrue(model.Weights.w1.Span.SequenceEqual(r.weights.w1));
            Assert.IsTrue(model.Weights.w2.Span.SequenceEqual(r.weights.w2));
            Assert.IsTrue(model.Weights.w3.Span.SequenceEqual(r.weights.w3));
            Assert.IsTrue( model.Weights.rms_final_weight.Span.SequenceEqual(r.weights.rms_final_weight));
            Assert.IsTrue( model.Weights.freq_cis_real.Span.SequenceEqual(r.weights.freq_cis_real));
            Assert.IsTrue( model.Weights.freq_cis_imag.Span.SequenceEqual(r.weights.freq_cis_imag));
            Assert.IsTrue(model.Weights.wcls.Span.SequenceEqual(r.weights.wcls));
        }

        [TestMethod]
        public void ReadVocab()
        {
            orgcode org = new orgcode();
            var r = org.LoadWeights(WEIGHT_FILE_PATH);
            Model model = new Model();
            model.LoadHeader(WEIGHT_FILE_PATH);
            model.LoadWeights(WEIGHT_FILE_PATH);

            var vocabs = org.ReadVocab(VOCAB_FILE_PATH, r.config.vocab_size);
            var t = new Tokenlizer(VOCAB_FILE_PATH, model.Header.Vocab_Size);
            Assert.IsTrue(vocabs.vocab.SequenceEqual(t.Vocab));
            Assert.IsTrue(vocabs.vocabScores.SequenceEqual(t.Scores));
        }
        [TestMethod]
        public void BPEEncode()
        {
            string prompt = "once upon a time";
            orgcode org = new orgcode();
            var r = org.LoadWeights(WEIGHT_FILE_PATH);
            Model model = new Model();
            model.LoadHeader(WEIGHT_FILE_PATH);
            model.LoadWeights(WEIGHT_FILE_PATH);
            var vocabs = org.ReadVocab(VOCAB_FILE_PATH, r.config.vocab_size);
            var t = new Tokenlizer(VOCAB_FILE_PATH, model.Header.Vocab_Size);

            var tt = org.BpeEncode(prompt, vocabs.vocab, vocabs.vocabScores, r.config.vocab_size, vocabs.maxTokenLength);
            var new_t = t.GetTokens(prompt);
            Assert.IsTrue(tt.tokens.Take(tt.nTokens).SequenceEqual(new_t));
        }
        [TestMethod]
        public void RMSNorm()
        {
            orgcode org = new orgcode();
            var r = org.LoadWeights(WEIGHT_FILE_PATH);
            Model model = new Model();
            model.LoadHeader(WEIGHT_FILE_PATH);
            model.LoadWeights(WEIGHT_FILE_PATH);
            Memory<float> o1 = new Memory<float>(new float[r.weights.rms_att_weight.Count]);
            r.weights.rms_att_weight.AsSpan().CopyTo(o1.Span);
            Memory<float> o2 = new Memory<float>(new float[model.Weights.rms_att_weight.Length]);
            model.Weights.rms_att_weight.Span.CopyTo(o2.Span);

            float[] o = randomF(r.config.hidden_dim);
            float[] x = randomF(r.config.dim);
            float[] w1 = randomF(r.config.dim);
            float[] w2 = new float[model.Header.Dims];
            randomFillF(o);
            randomFillF(x);
            randomFillF(w1);
            Array.Copy(w1, w2, w1.Length);
            org.Rmsnorm(o, x, w1, r.config.dim);
            MathHelper.RMSNorm(o, x, w2, model.Header.Dims);
            Console.WriteLine($"RMSNorm check: {w1.SequenceEqual(w2)}");
        }

        [TestMethod]
        public void MatMulTest()
        {
            orgcode org = new orgcode();
            int n = 100;
            int d = 90;
            float[] mx = randomF(n);
            float[] mw = randomF(n * d);
            float[] mo1 = new float[d];
            float[] mo2 = new float[d];
            org.Matmul(mo1, mx, mw, n, d);
            MathHelper.MatMul(mx, n, d, mw, mo2);
            //Assert.IsTrue(mo1.SequenceEqual(mo2));
            Assert.IsTrue(SequenceEqual(mo1, mo2, 3));
        }


        [TestMethod]
        public void SoftMaxTest()
        {
            orgcode org = new orgcode();
            float[] smax1 = randomF(100);
            float[] smax2 = new float[100];
            Array.Copy(smax1, smax2, smax1.Length);
            org.Softmax(smax1, 0, smax1.Length);
            MathHelper.SoftMax(smax2);
            Assert.IsTrue(SequenceEqual(smax1,smax2,6));
        }
        [TestMethod]
        public void AccumTest()
        {
            orgcode org = new orgcode();
            float[] v1 = randomF(100);
            float[] v3 = randomF(100);
            float[] v2 = new float[100];
            Array.Copy(v1, v2, v1.Length);
            org.Accum(v1, v3,v1.Length);
            MathHelper.Accum(v2, v3);
            Assert.IsTrue(v1.SequenceEqual(v2));
        }

        float[] randomF(int itemsCount)
        {
            float[] result = new float[itemsCount];
            randomFillF(result);
            return result;
        }
        void randomFillF(float[] data)
        {
            for (int i = 0; i < data.Length; i++)
            {
                data[i] = Random.Shared.NextSingle();
            }
        }

        bool SequenceEqual(IEnumerable<float> data1, IEnumerable<float> data2, int percision)
        {
            var data1_s = data1.Select(x => MathF.Round(x, percision));
            var data2_s = data2.Select(x => MathF.Round(x, percision));
            if (!data1_s.SequenceEqual(data2_s))
            {
                var diff = data1_s.Zip(data2_s).Zip(data1).Zip(data2).Where(v => v.First.First.First != v.First.First.Second);
                foreach (var item in diff)
                {
                    Debug.WriteLine($"{item.First.First.First} != {item.First.First.Second}   raw={item.First.Second},{item.Second}");
                }
            }
            return data1_s.SequenceEqual(data2_s);
        }


    }


}
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LLAMA2Sharp.UnitTest
{
    [TestClass]
    public class TransformTest
    {
        private string weightFilePath = @"d:\GithubRoot\llama2.cs\stories15M.bin";
        private string vocabFilePath = @"d:\GithubRoot\llama2.cs\tokenizer.bin";
        string TransformONEResult = "Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big, red ball in the sky. It was the sun! She wanted to touch it, but it was too high up. <0x0A>Lily asked her mommy, \"Can you help me touch the sun?\" <0x0A>Her mommy said, \"Sure, let's go outside and try to touch it together.\" <0x0A>They went outside and Lily's mommy held her hand and they touched the sun. It was so hot that they felt the heat on their skin. <0x0A>Lily said, \"It's so hot!\" <0x0A>Her mommy said, \"Yes, it is. But we can still touch it.\" <0x0A>Lily was happy that they could touch the sun and she could still play outside.";
        [TestMethod]
        public void CanRunTransform()
        {
            Model m = new Model();
            m.LoadHeader(weightFilePath);
            m.LoadWeights(weightFilePath);
            Tokenlizer t = new Tokenlizer(vocabFilePath,m.Header.Vocab_Size);
            RunContext context = RunContext.FromModelHeader(m.Header);
            m.Transform(1, 0, context);
        }

        [TestMethod]
        public void TransformONE()
        {
            StringBuilder sb = new StringBuilder();
            Model m = new Model();
            m.LoadHeader(weightFilePath);
            m.LoadWeights(weightFilePath);
            Tokenlizer t = new Tokenlizer(vocabFilePath, m.Header.Vocab_Size);
            RunContext context = RunContext.FromModelHeader(m.Header);
            var tokens = t.GetTokens(Consts.QUERY);
            int next = tokens[0];
            int token = 1;
            Stopwatch sw= Stopwatch.StartNew();
            for (int i = 0; i < m.Header.Seq_length; i++)
            {
                m.Transform(token,i,context);
                if (i<tokens.Length)
                {
                    next = tokens[i];
                }
                else
                {
                    next = MathHelper.ArgMax(context.logits.Span);
                    
                }
                if (next==1)
                {
                    break;
                }
                string tokenStr = token == 1 && t.Vocab[next][0] == ' ' ? t.Vocab[next].TrimStart() : t.Vocab[next];
                sb.Append(tokenStr);
                token = next;
                Debug.WriteLine($"{next} =>'{tokenStr}'");
            }
            sw.Stop();
            Debug.WriteLine(sw.Elapsed);
            Assert.AreEqual(TransformONEResult, sb.ToString());
        }
    }
}

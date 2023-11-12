using System;
using System.Buffers;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace LLAMA2Sharp
{
    public class Tokenlizer
    {
        public List<string> Vocab { get; private set; }
        public float[] Scores { get; private set; }
        public Tokenlizer(string filePath, int vocabSize)
        {
            using FileStream fs = File.OpenRead(filePath);
            using BinaryReader reader = new BinaryReader(fs);
            int maxTokenLength = reader.ReadInt32();
            Vocab = new List<string>(vocabSize);
            Scores = new float[vocabSize];
            var readBuffer = new byte[maxTokenLength].AsSpan();
            for (int i = 0; i < vocabSize; i++)
            {
                Scores[i] = reader.ReadSingle();
                int len = reader.ReadInt32();
                if (len > maxTokenLength)
                {
                    throw new InvalidOperationException("read token file error, token length is larger than maxTokenLength");
                }
                var tmpBuffer = readBuffer.Slice(0, len);
                reader.Read(tmpBuffer);
                Vocab.Add(Encoding.UTF8.GetString(tmpBuffer));
            }

        }

        public int[] GetTokens(string text)
        {
            int[] tokens = new int[text.Length];
            int tokenCount = 0;
            //first encode every individual byte in the input string
            StringBuilder sb = new StringBuilder();
            foreach (var c in text)
            {
                sb.Clear();
                sb.Append(c);
                var idx = Vocab.IndexOf(sb.ToString());
                if (idx == -1)
                {
                    throw new InvalidOperationException("embedding error,{c} not found in vocab");
                }
                tokens[tokenCount++] = idx;
            }
            while (true)
            {
                float bestScore = float.MinValue;
                int bestId = -1;
                int bestIdx = -1;

                for (int i = 0; i < tokenCount - 1; i++)
                {
                    // check if we can merge the pair (tokens[i], tokens[i+1])
                    sb.Clear();
                    sb.Append(Vocab[tokens[i]]);
                    sb.Append(Vocab[tokens[i + 1]]);

                    int id = Vocab.IndexOf(sb.ToString());
                    if (id != -1 && Scores[id] > bestScore)
                    {
                        // this merge pair exists in vocab! record its score and position
                        bestScore = Scores[id];
                        bestId = id;
                        bestIdx = i;
                    }
                }

                if (bestIdx == -1) // we couldn't find any more pairs to merge, so we're done
                {
                    break;
                }

                // merge the consecutive pair (bestIdx, bestIdx+1) into new token bestId
                tokens[bestIdx] = bestId;
                // delete token at position bestIdx+1, shift the entire sequence back 1
                for (int i = bestIdx + 1; i < tokenCount - 1; i++)
                {
                    tokens[i] = tokens[i + 1];
                }
                tokenCount--; // token length decreased
            }
            int[] result = new int[tokenCount];
            Array.Copy(tokens, result, tokenCount);
            return result;
        }
    }
}

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LLAMA2Sharp.New
{
    internal class NNLayerContext
    {
        
        public Memory<float> Input { get; set; }
        public Memory<float> Output { get;}

        public Memory<float> RMS_Att_Weight { get; private set; }

        public Memory<float> Q { get; private set; }
        public Memory<float> K { get; private set; }
        public Memory<float> V { get; private set; }

        public Memory<float> GetFreqCISReal(int pos)
        {

        }

        public Memory<float> GetFreqCISImg(int pos)
        {

        }
        public Memory<float> GetKeyCache(int pos)
        {

        }

        public Memory<float> GetValueCache(int pos)
        {

        }

        public Memory<float> AttentionBuffer { get; private set; }
        public Memory<float> AttentionResultBuffer { get; private set; } //原Xb注意力头结果缓冲区

        private Memory<float> xb2, hb, hb2;//internal buffer


    }
}

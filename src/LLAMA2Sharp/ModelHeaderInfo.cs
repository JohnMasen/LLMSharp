using System;
using System.Runtime.InteropServices;

namespace LLAMA2Sharp
{
    [StructLayout(LayoutKind.Sequential)]
    public struct ModelHeaderInfo
    {
        public int Dims; // transformer dimension
        public int Hidden_Dims; // for ffn layers
        public int N_layers; // number of layers
        public int N_heads; // number of query heads
        public int N_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
        public int v; // vocabulary size, usually 256 (byte-level)
        public int Seq_length; // max sequence length
        public bool SharedWeights => v > 0;
        /// <summary>
        /// vocabulary size,usually 256 (byte-level)
        /// </summary>
        public int Vocab_Size => Math.Abs(v);
        /// <summary>
        /// N_Layers * Dims
        /// </summary>
        public int LayersDim => N_layers * Dims;
        /// <summary>
        /// N_Layers * Dims * Dims
        /// </summary>
        public int LayersDim2 => LayersDim * Dims;
        /// <summary>
        /// N_Layers * Dims * Hidden_Dims
        /// </summary>
        public int LayersDimHiddenDim => LayersDim * Hidden_Dims;
        /// <summary>
        /// Dims / N_heads
        /// </summary>
        public int HeadSize => Dims / N_heads;
        /// <summary>
        /// N_heads / 2
        /// </summary>
        public int HeadSizeHalf => HeadSize / 2;
    }
}

using ILGPU.Algorithms.Optimization.Optimizers;
using ILGPU;
using Microsoft.VisualBasic.FileIO;
using System;
using System.Buffers;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Drawing;
using System.Linq;
using System.Numerics;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Text;
using System.Threading.Tasks;
using static System.Numerics.Tensors.TensorPrimitives;
namespace LLAMA2Sharp
{
    public static class MathHelper
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Multiply(this Span<float> source, Span<float> target)
        {
            TensorPrimitives.Multiply(source, target, source);
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void RMSNorm(Span<float> o, ReadOnlySpan<float> x, ReadOnlySpan<float> weight, int size)
        {
            // calculate sum of squares
            float ss = x.SelfDot();
            ss /= size;
            ss += 1e-5f;
            ss = 1.0f / MathF.Sqrt(ss);
            
            Multiply3(weight, x, ss, o);
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float SelfDot(this ReadOnlySpan<float> value)
        {
            return Dot(value, value);
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Multiply3(ReadOnlySpan<float> v1, ReadOnlySpan<float> v2, float v3, Span<float> target)
        {
            TensorPrimitives.Multiply(v1, v2, target);
            TensorPrimitives.Multiply(target, v3, target);
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void SILU(Span<float> value)
        {
            //    value[i] *= 1f / (1f + MathF.Exp(-value[i]));
            using var m = MemoryPool<float>.Shared.Rent(value.Length);
            var mem = m.Memory.Slice(0, value.Length);

            Sigmoid(value, mem.Span);
            TensorPrimitives.Multiply(value, mem.Span, value);//value*=1/(1+exp(-value))
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Accum(Span<float> value1, ReadOnlySpan<float> value2)
        {
            if (value1.Length!=value2.Length)
            {
                throw new ArgumentException("span1 should have same length to span2");
            }
            Add(value1, value2, value1);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void MatMul(ReadOnlySpan<float> input,int n,int d,ReadOnlySpan<float> weight,Span<float> output)
        {
            if (d * n != weight.Length)
            {
                throw new ArgumentException("weight length shoud equal d*n");
            }
            if (input.Length != n)
            {
                throw new ArgumentException("input should have same length to n");
            }
            if (output.Length != d)
            {
                throw new ArgumentException("output length should equal to d");
            }

            for (int i = 0; i < d; i++)
            {
                var weightBlock = weight.Slice(i * n, n);
                output[i] = Dot(input, weightBlock);
            }
        }

        public static int ArgMax(Span<float> probabilities) 
        {
            int maxIndex = 0;
            float maxValue = probabilities[0];
            for (int i = 1; i < probabilities.Length; i++)
            {
                if (probabilities[i]>maxValue)
                {
                    maxValue = probabilities[i];
                    maxIndex = i;
                }
            }
            return maxIndex;
        }

        public static void SoftMax(Span<float> data)
        {
            float max = data[0];
            float sum = 0;

            max = Max(data);
            Subtract(data, max, data);//data=data-max,special case for LLAMA2 softmax
            TensorPrimitives.SoftMax(data, data);

        }

        /// <summary>
        /// RoPE calculation
        /// </summary>
        /// <param name="value">source value, length must equal to dims</param>
        /// <param name="dims">dims</param>
        /// <param name="fr">real part for current position, length must equal to headsize/2</param>
        /// <param name="fi">image part for current position, length must equal to headsize/2</param>
        public static void RoPE(scoped in Span<float> value, scoped in ReadOnlySpan<float> fr,scoped in ReadOnlySpan<float> fi)
        {
            //new code
            var tData = new TensorSpan<float>(value, [value.Length],ReadOnlySpan<nint>.Empty);
            var tFr = new ReadOnlyTensorSpan<float>(fr, [fr.Length], ReadOnlySpan<nint>.Empty);
            var tFi = new ReadOnlyTensorSpan<float>(fi, [fi.Length], ReadOnlySpan<nint>.Empty);
            //Tensor.Create<float>([1f, 2f], [2]);
            var tfr1=Tensor.CreateUninitialized<float>([fr.Length]);
            tFr.CopyTo(tfr1.AsTensorSpan());
            var tfi1 = Tensor.CreateUninitialized<float>([fi.Length]);
            tFi.CopyTo(tfi1.AsTensorSpan());


            var tF = MathHelper.StackAlongDimension<float>(1, tfr1, tfi1);


            var data=MemoryMarshal.Cast<float, Vector2>(value);
            using var mo = MemoryPool<Vector2>.Shared.Rent(fr.Length*2);
            var f1 = mo.Memory.Slice(0, fr.Length).Span;
            var f2 = mo.Memory.Slice(fr.Length, fr.Length).Span;
            for (int i = 0; i < fr.Length; i++)
            {
                f1[i] = new Vector2(fr[i], fi[i]);
                f2[i] = new Vector2(fr[i], -fi[i]);
            }

            //current code
            for (int j = 0; j < value.Length; j += 2)
            {
                float q0 = value[j];
                float q1 = value[j + 1];
                float fcr = fr[j/2 % fr.Length];
                float fci = fi[j/2 % fi.Length]; 
                value[j] = q0 * fcr - q1 * fci;
                value[j + 1] = q0 * fci + q1 * fcr;
            }
        }



        public static void DumpSpan(ReadOnlySpan<float> value)
        {
            Debug.WriteLine("Begin dump---");
            foreach (var item in value)
            {
                Debug.WriteLine(item);
            }
            Debug.WriteLine("End dump---");
            Debugger.Break();
            
        }

        public static int Sample(ReadOnlySpan<float> probabilities,int seed=-1)
        {

            float r = seed==-1? Random.Shared.NextSingle():new Random(seed).NextSingle();

            int count = probabilities.Length / Vector<float>.Count;
            int remaining = probabilities.Length % Vector<float>.Count;
            ReadOnlySpan<Vector<float>> s1 = MemoryMarshal.Cast<float, Vector<float>>(probabilities);
            float sum = 0;
            for (int i = 0; i < count; i++)
            {
                float last = sum;
                sum += Vector.Sum(s1[i]);
                if (sum>r)
                {
                    for (int j = 0; j < Vector<float>.Count; j++)
                    {
                        last += s1[i][j];
                        if (last>r)
                        {
                            return i * Vector<float>.Count+j;
                        }
                    }
                }
            }
            for (int i = 0; i < remaining; i++)
            {
                int pos = probabilities.Length - remaining;
                sum += probabilities[pos + i];
                if (sum>r)
                {
                    return pos;
                }
            }

            return probabilities.Length - 1;
        }

        private static Tensor<T> StackAlongDimension<T>(int dimension,params Tensor<T>[] tensors)
        {

            if (dimension < 0)
            {
                dimension = tensors[0].Rank - dimension;
            }

            Tensor<T>[] array = new Tensor<T>[tensors.Length];
            for (int j = 0; j < tensors.Length; j++)
            {
                array[j] = tensors[j].Unsqueeze(dimension);
            }
            return Tensor.ConcatenateOnDimension<T>(dimension, array);
        }
    }
}

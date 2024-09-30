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

namespace LLAMA2Sharp
{
    public static class MathHelper
    {

        public static void Multiply(this Span<float> source, Span<float> target)
        {
            if (source.Length!=target.Length)
            {
                throw new ArgumentException("[DOT]Span length is not same");
            }
            int count = source.Length / Vector<float>.Count;
            int remaining = source.Length % Vector<float>.Count;
            Span<Vector<float>> v1s = MemoryMarshal.Cast<float, Vector<float>>(source);
            ReadOnlySpan<Vector<float>> v2s = MemoryMarshal.Cast<float, Vector<float>>(target);
            for (int i = 0; i < count; i++)
            {
                v1s[i] *= v2s[i];
            }
            for (int i = 1; i <= remaining; i++)
            {
                source[^i] *= target[^i];
            }
        }
        public static void RMSNorm(Span<float> o, ReadOnlySpan<float> x, ReadOnlySpan<float> weight, int size)
        {
            // calculate sum of squares
            float ss = x.SelfDot();
            ss /= size;
            ss += 1e-5f;
            ss = 1.0f / MathF.Sqrt(ss);

            // normalize and scale
            //for (int j = 0; j < size; j++) o.Span[j] = weight.Span[j] * (ss * x.Span[j]);
            Multiply3(weight, x, ss, o);
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float SelfDot(this ReadOnlySpan<float> value)
        {
            return TensorPrimitives.Dot(value, value);
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Multiply3(ReadOnlySpan<float> v1, ReadOnlySpan<float> v2, float v3, Span<float> target)
        {
            //int count = v1.Length / Vector<float>.Count;
            //int remaining = v1.Length % Vector<float>.Count;
            //ReadOnlySpan<Vector<float>> v1s = MemoryMarshal.Cast<float, Vector<float>>(v1);
            //ReadOnlySpan<Vector<float>> v2s = MemoryMarshal.Cast<float, Vector<float>>(v2);
            //Span<Vector<float>> outputs = MemoryMarshal.Cast<float, Vector<float>>(target);
            //for (int i = 0; i < count; i++)
            //{
            //    outputs[i] = v1s[i] * v2s[i] * v3;
            //}
            //for (int i = 1; i <= remaining; i++)
            //{
            //    target[^i] = v1[^i] * v2[^i] * v3;
            //}
            TensorPrimitives.Multiply(v1, v2, target);
            TensorPrimitives.Multiply(target, v3, target);
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void SILU(Span<float> value)
        {
            //for (int i = 0; i < value.Length; i++)
            //{
            //    value[i] *= 1f / (1f + MathF.Exp(-value[i]));
            //}
            using var m = MemoryPool<float>.Shared.Rent(value.Length);
            var mem = m.Memory.Slice(0, value.Length);

            TensorPrimitives.Sigmoid(value, mem.Span);
            TensorPrimitives.Multiply(value, mem.Span, value);//value*=1/(1+exp(-value))
        }

        //[MethodImpl(MethodImplOptions.AggressiveInlining)]
        //public static void WithSpan(ReadOnlySpan<float> span ,Action<Vector<float>> action,Action<float> remainAction) 
        //{
        //    int count = span.Length / Vector<float>.Count;
        //    int remaining = span.Length % Vector<float>.Count;
        //    ReadOnlySpan<Vector<float>> s = MemoryMarshal.Cast<float, Vector<float>>(span);
        //    for (int i = 0; i < count; i++)
        //    {
        //        action(s[i]);
        //    }
        //    for (int i = 1; i <= remaining; i++)
        //    {
        //        remainAction(span[^i]);
        //    }
        //}

        //[MethodImpl(MethodImplOptions.AggressiveInlining)]
        //public static void WithSpanWB(Span<float> span, Func<Vector<float>,Vector<float>> func, Func<float,float> remainFunc)
        //{
        //    int count = span.Length / Vector<float>.Count;
        //    int remaining = span.Length % Vector<float>.Count;
        //    Span<Vector<float>> s = MemoryMarshal.Cast<float, Vector<float>>(span);
        //    for (int i = 0; i < count; i++)
        //    {
        //        s[i]=func(s[i]);
        //    }
        //    for (int i = 1; i <= remaining; i++)
        //    {
        //        span[^i]= remainFunc(span[^i]);
        //    }
        //}

        //[MethodImpl(MethodImplOptions.AggressiveInlining)]
        //public static void WithSpan(ReadOnlySpan<float> span1, ReadOnlySpan<float> span2, Action<Vector<float>,Vector<float>> action, Action<float,float> remainAction)
        //{
        //    if (span1.Length!=span2.Length)
        //    {
        //        throw new ArgumentException("span1 should have same length as span2");
        //    }
        //    int count = span1.Length / Vector<float>.Count;
        //    int remaining = span1.Length % Vector<float>.Count;
        //    ReadOnlySpan<Vector<float>> s1 = MemoryMarshal.Cast<float, Vector<float>>(span1);
        //    ReadOnlySpan<Vector<float>> s2 = MemoryMarshal.Cast<float, Vector<float>>(span2);
        //    for (int i = 0; i < count; i++)
        //    {
        //        action(s1[i], s2[i]);
        //    }
        //    for (int i = 1; i <= remaining; i++)
        //    {
        //        remainAction(span1[^i], span2[^i]);
        //    }
        //}

        //[MethodImpl(MethodImplOptions.AggressiveInlining)]
        //public static void WithSpanWB(Span<float> span1, ReadOnlySpan<float> span2, Func<Vector<float>, Vector<float>,Vector<float>> func, Func<float,float, float> remainFunc)
        //{
        //    if (span1.Length != span2.Length)
        //    {
        //        throw new ArgumentException("span1 should have same length as span2");
        //    }
        //    int count = span1.Length / Vector<float>.Count;
        //    int remaining = span1.Length % Vector<float>.Count;
        //    Span<Vector<float>> s1 = MemoryMarshal.Cast<float, Vector<float>>(span1);
        //    ReadOnlySpan<Vector<float>> s2 = MemoryMarshal.Cast<float, Vector<float>>(span2);
        //    for (int i = 0; i < count; i++)
        //    {
        //        s1[i]=func(s1[i], s2[i]);
        //    }
        //    for (int i = 1; i <= remaining; i++)
        //    {
        //        span1[^i]=remainFunc(span1[^i], span2[^i]);
        //    }
        //}

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Accum(Span<float> value1, ReadOnlySpan<float> value2)
        {
            if (value1.Length!=value2.Length)
            {
                throw new ArgumentException("span1 should have same length to span2");
            }
            //int count = value1.Length / Vector<float>.Count;
            //int remaining = value2.Length % Vector<float>.Count;
            //Span<Vector<float>> s1 = MemoryMarshal.Cast<float, Vector<float>>(value1);
            //ReadOnlySpan<Vector<float>> s2 = MemoryMarshal.Cast<float, Vector<float>>(value2);
            //for (int i = 0; i < count; i++)
            //{
            //    s1[i] += s2[i];
            //}
            //for (int i = 1; i <= remaining; i++)
            //{
            //    value1[^i] += value2[^i];
            //}
            TensorPrimitives.Add(value1, value2, value1);
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
            //for (int i = 0; i < d; i++)
            //{
            //    var weightChunk = weight.Slice(i * n, n);
            //    float f = 0;
            //    WithSpan(weightChunk, input, (v1, v2) =>
            //    {
            //        f += Vector.Dot(v1, v2);
            //    },
            //    (v1, v2) =>
            //    {
            //        f += v1 * v2;
            //    });
            //    output[i] = f;
            //}


            //for (int i = 0; i < d; i++)
            //{
            //    float val = 0.0f;
            //    for (int j = 0; j < n; j++) val += weight[i * n + j] * input[j];
            //    output[i] = val;
            //}


            for (int i = 0; i < d; i++)
            {
                var weightBlock = weight.Slice(i * n, n);
                output[i] = TensorPrimitives.Dot(input, weightBlock);
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
            //Vector<float> maxV = new Vector<float>(float.MinValue);
            ////find max value
            //WithSpan(data, v =>
            //{
            //    maxV = Vector.Max(v, maxV);
            //},
            //v =>
            //{
            //    max = MathF.Max(max, v);
            //});
            //for (int i = 0; i < Vector<float>.Count; i++)
            //{
            //    max = MathF.Max(maxV[i], max);
            //}
            ////calculate exp
            //for (int i = 0; i < data.Length; i++)
            //{
            //    data[i] = MathF.Exp(data[i] - max);
            //}
            ////calculate sum
            //WithSpan(data, v =>
            //{
            //    sum += Vector.Sum(v);
            //}, v =>
            //{
            //    sum += v;
            //});
            ////Vector<float> sumV = new Vector<float>(sum);
            //float RE_Sum = 1/sum;
            ////calculate result
            //WithSpanWB(data, v =>
            //{
            //    return Vector.Multiply(v, RE_Sum);
            //}, v =>
            //{
            //    return v * RE_Sum;
            //});

            //float max = data[0];
            //for (int i = 1; i < data.Length; i++)
            //    if (data[i] > max)
            //        max = data[i];
            //exp and sum
            //float sumVal = 0.0f;
            //for (int i = 0; i < data.Length; i++)
            //{
            //    //data[i] = (float)Math.Exp(data[i] - max);
            //    sumVal += data[i];
            //}

            //normalize
            //for (int i = 0; i < data.Length; i++) data[i] /= sum;

            max = TensorPrimitives.Max(data);
            TensorPrimitives.Subtract(data, max, data);//data=data-max
            TensorPrimitives.SoftMax(data, data);

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
                            return i * Vector<float>.Count;
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
    }
}

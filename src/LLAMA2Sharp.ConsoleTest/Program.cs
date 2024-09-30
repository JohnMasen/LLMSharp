// See https://aka.ms/new-console-template for more information
using LLAMA2Sharp;
using LLAMA2Sharp.ConsoleTest;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

string weightFilePath = @"d:\GithubRoot\llama2.cs\stories15M.bin";
string vocabFilePath = @"d:\GithubRoot\llama2.cs\tokenizer.bin";
string prompt = "Once upon a time";
Console.WriteLine("Hello, World!");
orgcode org = new orgcode();
var r=org.LoadWeights(weightFilePath);
var c = r.config;

Model model = new Model();
model.LoadHeader(weightFilePath);
model.LoadWeights(weightFilePath);

Console.WriteLine(dataCheck("embedding table",model.Weights.token_embedding_table,r.weights.token_embedding_table));
Console.WriteLine(dataCheck("rms_att_weight", model.Weights.rms_att_weight, r.weights.rms_att_weight));
Console.WriteLine(dataCheck("rms_ffn_weight",model.Weights.rms_ffn_weight,r.weights.rms_ffn_weight));
Console.WriteLine(dataCheck("wq", model.Weights.wq, r.weights.wq));
Console.WriteLine(dataCheck("wk", model.Weights.wk, r.weights.wk));
Console.WriteLine(dataCheck("wv", model.Weights.wv, r.weights.wv));
Console.WriteLine(dataCheck("wo", model.Weights.wo, r.weights.wo));
Console.WriteLine(dataCheck("w1", model.Weights.w1, r.weights.w1));
Console.WriteLine(dataCheck("w2", model.Weights.w2, r.weights.w2));
Console.WriteLine(dataCheck("w3", model.Weights.w3, r.weights.w3));
Console.WriteLine(dataCheck("rms_final_weight", model.Weights.rms_final_weight, r.weights.rms_final_weight));
Console.WriteLine(dataCheck("freq_cis_real", model.Weights.freq_cis_real, r.weights.freq_cis_real));
Console.WriteLine(dataCheck("freq_cis_imag", model.Weights.freq_cis_imag, r.weights.freq_cis_imag));
Console.WriteLine(dataCheck("wcls", model.Weights.wcls, r.weights.wcls));

dumpHeader(model.Header);


var vocabs=org.ReadVocab(vocabFilePath, r.config.vocab_size);
var t = new Tokenlizer(vocabFilePath, model.Header.Vocab_Size);
Console.WriteLine($"vocab check :{vocabs.vocab.SequenceEqual(t.Vocab)}");
Console.WriteLine($"vocabScore check :{vocabs.vocabScores.SequenceEqual(t.Scores)}");

var tt=org.BpeEncode(prompt, vocabs.vocab, vocabs.vocabScores, c.vocab_size, vocabs.maxTokenLength);
var new_t = t.GetTokens(prompt);
Console.WriteLine($"embedding check: {tt.tokens.Take(tt.nTokens).SequenceEqual(new_t)}");


Memory<float> o1 = new Memory<float>(new float[r.weights.rms_att_weight.Count]);
r.weights.rms_att_weight.AsSpan().CopyTo(o1.Span);
Memory<float> o2 = new Memory<float>(new float[model.Weights.rms_att_weight.Length]);
model.Weights.rms_att_weight.Span.CopyTo(o2.Span);

float[] o = new float[r.config.hidden_dim];
float[] x = new float[r.config.dim];
float[] w1 = new float[r.config.dim];
float[] w2 = new float[model.Header.Dims];
randomFillF(o);
randomFillF(x);
randomFillF(w1);
Array.Copy(w1, w2, w1.Length);
org.Rmsnorm(o, x, w1, r.config.dim);
MathHelper.RMSNorm(o, x, w2, model.Header.Dims);
Console.WriteLine($"RMSNorm check: {w1.SequenceEqual(w2)}");

Stopwatch sw = Stopwatch.StartNew();
int count = 10000;
for (int i = 0; i < count; i++)
{
    org.Rmsnorm(o, x, w1, r.config.dim);
}
sw.Stop();
Console.WriteLine($"org Rmsnorm={sw.ElapsedMilliseconds}ms");

sw.Restart();
for (int i = 0; i < count; i++)
{
    MathHelper.RMSNorm(o, x, w2, model.Header.Dims);
}
sw.Stop();
Console.WriteLine($"New Rmsnorm={sw.ElapsedMilliseconds}ms");

int n = 100;
int d = 90;
float[] mx = randomF(n);
float[] mw = randomF(n * d); 
float[] mo1 = new float[d];
float[] mo2 = new float[d];
org.Matmul(mo1, mx, mw, n, d);
MathHelper.MatMul(mx, n, d, mw, mo2);
Console.WriteLine($"MatMul check: {lowpercisionSequenceCheck(mo1,mo2,2)}");

float[] smax1 = randomF(100);
float[] smax2 = new float[100];
Array.Copy(smax1, smax2, smax1.Length);
org.Softmax(smax1,0,smax1.Length);
MathHelper.SoftMax(smax2);
Console.WriteLine($"Softmax check: {lowpercisionSequenceCheck(smax1,smax2,2)}");





//Console.ReadLine();

string dataCheck(string name,ReadOnlyMemory<float> value1, ReadOnlySpan<float> value2)
{
    return $"{name} check : {value1.Span.SequenceEqual(value2)}";
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

bool lowpercisionSequenceCheck(IEnumerable<float>data1,IEnumerable<float> data2, int percision)
{
    var data1_s = data1.Select(x =>  MathF.Round(x,percision));
    var data2_s = data2.Select(x =>  MathF.Round(x, percision));
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

void dumpHeader(ModelHeaderInfo header)
{
    Console.WriteLine("-------------Header info Start-------------");
    foreach (var f in header.GetType().GetFields())
    {
        Console.WriteLine($"[F]{f.Name}= {f.GetValue(header)}");
    }
    foreach (var p in header.GetType().GetProperties())
    {
        Console.WriteLine($"[P]{p.Name}= {p.GetValue(header)}");
    }
    
    Console.WriteLine("-------------Header info End-------------");
}
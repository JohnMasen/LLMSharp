using ILGPU.IR;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LLAMA2Sharp.New
{
    public interface ILLMModel
    {
        
        public async Task Step(int token,int pos)
        {

        }
    }
}

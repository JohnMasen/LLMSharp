using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LLAMA2Sharp.New
{
    public interface IWeightManager
    {
        public Task<Memory<float>> GetLayer(int layerId);


        public Task<Memory<float>> GetAttention(int attentionId);

        

    }
}

using System;

namespace ML.Libary
{
    public class NeuralNetwork
    {
        public int[][] Weights { get; set; }
        public int[] Bias { get; set; }
        private int[] Buffer { get; set; }
        public int Layer { get { return _layer; } set { _layer = value; } }
        private int _layer = 0;

        ///<summary>
        /// Process actual layer.
        ///</summary>
        ///<param name="layer"></param>
        ///<returns>Outputs processeds.</returns>
        private int[] ProccessLayer(int[] layer, int[] values)
        {
            int[] outputs = new int[layer.Length];

            //Calcule neurons values in layer.
            for (int i = 0; i < layer.Length; i++)
            {
                //Multiply the input values with neuron layer value. 
                for (int j = 0; j < values.Length; j++)
                {
                    outputs[i] += values[i] * layer[i];
                }

                //Add Bias in neuron value
                outputs[i] += Bias[i];
            }
            return outputs;
        }

        public int[] ComputeInput(int[] inputs)
        {
            Buffer = ProccessLayer(Weights[0], inputs);
            for (int i = 1; i < Weights.Length; i++)
            {
                Layer = i;
                Buffer = ProccessLayer(Weights[i], Buffer);
            }
            return Buffer;
        }
    }
}

using System;
using System.Collections.Generic;
using System.Linq;
using Newtonsoft.Json;

[Serializable]
public class NeuralNetworkDto
{
    public List<string[][]> Weights { get; set; }
    public List<string> Biases { get; set; }

    [JsonConstructor]
    public NeuralNetworkDto(List<string[][]> weights, List<string> biases)
    {
        Weights = weights;
        Biases = biases;
    }

    public NeuralNetworkDto(NeuralNetwork neuralNetwork)
    {
        var biasesString = neuralNetwork.Biases.Select(b => "" + b).ToList();
        var weightsString = neuralNetwork.Weights
            .Select(w => w
                .ToMultiDimensionalArray()
                .Select(row => row.Select(col => "" + col).ToArray())
                .ToArray())
            .ToList();
        
        Weights = weightsString;
        Biases = biasesString;
    }

    public List<Matrix> GetWeight()
    {
        return Weights
            .Select(w => w.Select(row => row.Select(double.Parse).ToArray()).ToArray())
            .Select(w => new Matrix(w.Length, w[0].Length, w))
            .ToList();
    }

    public List<double> GetBiases()
    {
        return Biases.Select(double.Parse).ToList();
    }
}
using System;
using System.Collections.Generic;
using System.Linq;
using Newtonsoft.Json;
using UnityEngine;
using Random = UnityEngine.Random;

[Serializable]
public class NeuralNetwork : ICloneable
{
    public Matrix InputLayer { get; private set; }
    public List<Matrix> HiddenLayers { get; private set; }
    public Matrix OutputLayer { get; private set; }
    public List<Matrix> Weights { get; private set; }
    public List<double> Biases { get; private set; }
    public Func<double, double>[] ActivationsFunctions { get; private set; }

    [field: NonSerialized] [JsonIgnore] public double Fitness { get; set; }

    [JsonConstructor]
    private NeuralNetwork(int inputsAmount, int outputsAmount, Func<double, double>[] functions)
    {
        InputLayer = new Matrix(1, inputsAmount);
        HiddenLayers = new List<Matrix>();
        OutputLayer = new Matrix(1, outputsAmount);
        Weights = new List<Matrix>();
        Biases = new List<double>();
        ActivationsFunctions = functions;
    }
    
    public void Init(int inputsAmount, int[] neuronsInHiddenLayersAmount, int outputsAmount,
        Func<double, double>[] activationsFunctions)
    {
        Validate(neuronsInHiddenLayersAmount, activationsFunctions.Length, outputsAmount);
        ActivationsFunctions = activationsFunctions;

        GenerateWeightsMatrices(inputsAmount, neuronsInHiddenLayersAmount);

        var weightsLastHiddenLayerToOutputLayer =
            new Matrix(neuronsInHiddenLayersAmount[neuronsInHiddenLayersAmount.Length - 1], outputsAmount);
        Weights.Add(weightsLastHiddenLayerToOutputLayer);
        Biases.Add(Random.Range(-1f, 1f));

        GenerateWeightsValues();
    }

    private void GenerateWeightsMatrices(int inputsAmount, int[] neuronsInHiddenLayersAmount)
    {
        for (var neuronsCount = 0; neuronsCount < neuronsInHiddenLayersAmount.Length; neuronsCount++)
        {
            var hiddenLayer = new Matrix(1, neuronsInHiddenLayersAmount[neuronsCount]);
            HiddenLayers.Add(hiddenLayer);
            Biases.Add(Random.Range(-1f, 1f));

            if (neuronsCount == 0)
            {
                var weightsInputToHiddenLayer1 = new Matrix(inputsAmount, neuronsInHiddenLayersAmount[neuronsCount]);
                Weights.Add(weightsInputToHiddenLayer1);
            }
            else
            {
                var weightsHliToNextHl = new Matrix(neuronsInHiddenLayersAmount[neuronsCount - 1],
                    neuronsInHiddenLayersAmount[neuronsCount]);
                Weights.Add(weightsHliToNextHl);
            }
        }
    }

    public double[] Run(double[] inputs)
    {
        if (inputs.Length != InputLayer.ColsCount)
            throw new ArgumentException($"Wrong inputs size! Neural network has {InputLayer.RowsCount} inputs.");

        for (var i = 0; i < inputs.Length; i++)
        {
            InputLayer[0, i] = inputs[i];
        }

        InputLayer = InputLayer.PointwiseTanh();

        HiddenLayers[0] = (InputLayer * Weights[0] + Biases[0]).PointwiseTanh();

        for (var i = 1; i < HiddenLayers.Count; i++)
        {
            HiddenLayers[i] = (HiddenLayers[i - 1] * Weights[i] + Biases[i]).PointwiseTanh();
        }

        OutputLayer = (HiddenLayers[HiddenLayers.Count - 1] * Weights[Weights.Count - 1] + Biases[Biases.Count - 1])
            .PointwiseTanh();

        var output = OutputLayer.ToOneDimensionalArray();
        for (var i = 0; i < ActivationsFunctions.Length; i++)
        {
            output[i] = ActivationsFunctions[i](output[i]);
        }

        return output;
    }

    private static void Validate(int[] neuronsInHiddenLayersAmount, int activationsFunctionsAmount, int outputsAmount)
    {
        if (neuronsInHiddenLayersAmount.Length < 1)
            throw new ArgumentException("Neural Network must have at least 1 hidden layer!");
        if (activationsFunctionsAmount != outputsAmount)
            throw new ArgumentException("Number of activations functions must be equal to count of outputs!");
    }

    private void GenerateWeightsValues()
    {
        for (var weightIndex = 0; weightIndex < Weights.Count; weightIndex++)
        for (var rowI = 0; rowI < Weights[weightIndex].RowsCount; rowI++)
        for (var colJ = 0; colJ < Weights[weightIndex].ColsCount; colJ++)
            Weights[weightIndex][rowI, colJ] = Random.Range(-1f, 1f);
    }

    public object Clone()
    {
        var clone = Of(InputLayer.ColsCount, OutputLayer.ColsCount, ActivationsFunctions);
        var cloneWeights = new List<Matrix>();

        foreach (var weight in Weights)
        {
            var currentWeight = new Matrix(weight.RowsCount, weight.ColsCount);

            for (var rowI = 0; rowI < currentWeight.RowsCount; rowI++)
            {
                for (var colJ = 0; colJ < currentWeight.ColsCount; colJ++)
                {
                    currentWeight[rowI, colJ] = weight[rowI, colJ];
                }
            }

            cloneWeights.Add(currentWeight);
        }

        var cloneBiases = new List<double>(Biases);

        clone.Weights = cloneWeights;
        clone.Biases = cloneBiases;

        var neuronsInHiddenLayers = HiddenLayers.Select(layer => layer.ColsCount).ToArray();
        clone.InitializeHiddenLayers(neuronsInHiddenLayers);
        
        return clone;
    }
    
    public void InitializeHiddenLayers (int[] hiddenNeuronCount)
    {
        InputLayer.Clear();
        HiddenLayers.Clear();
        OutputLayer.Clear();

        foreach (var neuronsCount in hiddenNeuronCount)
        {
            var newHiddenLayer = new Matrix(1, neuronsCount);
            HiddenLayers.Add(newHiddenLayer);
        }
    }


    public static NeuralNetwork Of(int inputsAmount, int outputsAmount, Func<double, double>[] functions)
    {
        return new NeuralNetwork(inputsAmount, outputsAmount, functions);
    }
    
    public static NeuralNetwork Of(NeuralNetworkDto dto, int inputsAmount, int[] hiddenNeuronCount, int outputsAmount, Func<double, double>[] functions)
    {
        var neuralNetwork = new NeuralNetwork(inputsAmount, outputsAmount, functions)
        {
            Weights = dto.GetWeight(), Biases = dto.GetBiases()
        };
        neuralNetwork.InitializeHiddenLayers(hiddenNeuronCount);
        return neuralNetwork;
    }
}
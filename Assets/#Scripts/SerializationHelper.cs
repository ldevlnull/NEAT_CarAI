using System;
using System.IO;
using System.Text;
using Newtonsoft.Json;
using UnityEngine;

public static class SerializationHelper
{
    private const string StatsPathPrefix = @"Assets\logs\Stats_";
    private const string SerializationDirectory = @"Assets\logs\SavedNetworks";
    private const string SerializationNeatDirectory = @"Assets\logs\SavedNeats";

    private static readonly JsonSerializer Serializer = new JsonSerializer();

    public static void SerializeStats(int currentGeneration, NeuralNetwork[] population, ref string statsFileName)
    {
        if (statsFileName == null || statsFileName.Equals(""))
            statsFileName = $"{StatsPathPrefix}{DateTime.Now.ToFileTime()}.csv";
        if (!File.Exists(statsFileName))
        {
            Directory.CreateDirectory(StatsPathPrefix);
            var newFile = File.CreateText(statsFileName);
            newFile.Write("Generation,Genome,Fitness\n\r");
            newFile.Close();
        }

        var text = new StringBuilder();
        for (var genome = 0; genome < population.Length; genome++)
        {
            text.Append($"{currentGeneration},{genome},{population[genome].Fitness}\n");
        }

        using (var file = File.AppendText(statsFileName))
        {
            file.Write(text.ToString());
        }
    }

    public static void SerializeNeuralNetwork(NeuralNetwork neuralNetwork, double fitness)
    {
        if (!Directory.Exists("logs"))
        {
            Directory.CreateDirectory("logs");
        }

        var path = GenerateNeuralNetworkFileName(fitness);
        using (var file = File.CreateText(path))
        {
            var dto = new NeuralNetworkDto(neuralNetwork);
            Serializer.Serialize(file, dto);
        }
    }

    public static NeuralNetwork DeserializeNeuralNetwork(CarAI carAI)
    {
        if (carAI.jsonFileNeuralNetwork == null)
        {
            throw new NullReferenceException("File path cannot be null if you want to deserialize network!");
        }

        var deserializeObject = JsonConvert.DeserializeObject<NeuralNetworkDto>(carAI.jsonFileNeuralNetwork.text);
        return NeuralNetwork.Of(deserializeObject, carAI.inputsAmount, carAI.neuronsInHiddenLayerCount,
            carAI.outputsAmount, carAI.ActivationsFunctions);
    }

    public static void SerializeNeat(NEAT_Manager neatManager)
    {
        var path = GenerateNeatFileName();
        Debug.Log(path);
        using (var file = File.CreateText(path))
        {
            var dto = new NeatDto(neatManager);
            Serializer.Serialize(file, dto);
        }
    }
    
    public static NEAT_Manager DeserializeNeat(CarAI carAI)
    {
        if (carAI.jsonFileNeat == null)
        {
            throw new NullReferenceException("File path cannot be null if you want to deserialize network!");
        }

        var deserializedObject = JsonConvert.DeserializeObject<NeatDto>(carAI.jsonFileNeat.text);
        return NeatDto.CreateNeat(deserializedObject);
    }

    private static string GenerateNeuralNetworkFileName(double fitness)
    {
        var fitnessStr = ("" + fitness).Split('.')[0];
        var dir = $"{SerializationDirectory}\\{fitnessStr}";
        if (!Directory.Exists(dir))
            Directory.CreateDirectory(dir);
        return $"{dir}\\NeuronNetwork_{DateTime.Now.ToFileTime()}.json";
    }

    private static string GenerateNeatFileName()
    {
        if (!Directory.Exists(SerializationNeatDirectory))
            Directory.CreateDirectory(SerializationNeatDirectory);

        return $"{SerializationNeatDirectory}\\Neat_{DateTime.Now.ToFileTime()}.json";
    }
}
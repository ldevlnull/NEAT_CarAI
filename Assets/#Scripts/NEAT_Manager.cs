using System;
using System.Collections.Generic;
using UnityEngine;
using Random = UnityEngine.Random;

// ReSharper disable once InconsistentNaming
public class NEAT_Manager : MonoBehaviour, IConfigurable
{
    private string _generatedStatsNameFile = "";

    [Header("NEAT Control")] [SerializeField]
    public int initialPopulation;

    [SerializeField] [Range(0f, 1f)] public double mutationChance;
    [SerializeField] public int fitnessMultiplier = 10;

    [Header("Crossover Control")] [SerializeField]
    public int bestAgentSelection = 8;

    [SerializeField] public int worstAgentSelection = 3;
    [SerializeField] public int numberToCrossover;
    [SerializeField] [Range(0, 1)] public double crossoverChance = 0.5;

    [Header("Public View")] [SerializeField]
    public int currentGeneration = 1;

    [SerializeField] public int currentGenome = 1;
    [SerializeField] public double bestFitness;

    public List<int> genesPool = new List<int>();
    public int naturallySelected;

    public NeuralNetwork[] population;

    public int inputsAmount;
    public int outputsAmount;
    public Func<double, double>[] activationFunctions;
    public int[] neuronsInHiddenLayerCount;

    public NEAT_Manager(int initialPopulation, double mutationChance, int fitnessMultiplier, int bestAgentSelection,
        int worstAgentSelection, int numberToCrossover, double crossoverChance, int currentGeneration,
        int currentGenome, double bestFitness, int naturallySelected, NeuralNetwork[] population, int inputsAmount,
        int outputsAmount, Func<double, double>[] activationFunctions, int[] neuronsInHiddenLayerCount)
    {
        _generatedStatsNameFile = "";
        this.initialPopulation = initialPopulation;
        this.mutationChance = mutationChance;
        this.fitnessMultiplier = fitnessMultiplier;
        this.bestAgentSelection = bestAgentSelection;
        this.worstAgentSelection = worstAgentSelection;
        this.numberToCrossover = numberToCrossover;
        this.crossoverChance = crossoverChance;
        this.currentGeneration = currentGeneration;
        this.currentGenome = currentGenome;
        this.bestFitness = bestFitness;
        this.naturallySelected = naturallySelected;
        this.population = population;
        this.inputsAmount = inputsAmount;
        this.outputsAmount = outputsAmount;
        this.activationFunctions = activationFunctions;
        this.neuronsInHiddenLayerCount = neuronsInHiddenLayerCount;
        genesPool = new List<int>();
    }

    public void PostConfigStart(CarAI carAI)
    {
        if (!carAI.isMaster) return;
        
        GUIHelper.AddToDisplay("Best fitness", () => bestFitness);
        GUIHelper.AddToDisplay("Generation", () => currentGeneration);
        GUIHelper.AddToDisplay("Genome", () => currentGenome);
        inputsAmount = carAI.inputsAmount;
        outputsAmount = carAI.outputsAmount;
        activationFunctions = carAI.ActivationsFunctions;
        neuronsInHiddenLayerCount = carAI.neuronsInHiddenLayerCount;

        CreatePopulation(carAI);
    }

    protected void CreatePopulation(CarAI carAI)
    {
        population = new NeuralNetwork[initialPopulation];
        RandomizePopulation(population, 0);
        ResetToCurrentGenome(carAI);
    }

    private void RandomizePopulation(NeuralNetwork[] newPopulation, int startingIndex)
    {
        while (startingIndex < initialPopulation)
        {
            newPopulation[startingIndex] =
                NeuralNetwork.Of(inputsAmount, outputsAmount, activationFunctions);
            newPopulation[startingIndex++].Init(inputsAmount, neuronsInHiddenLayerCount,
                outputsAmount, activationFunctions);
        }
    }

    private void ResetToCurrentGenome(CarAI carAI)
    {
        carAI.ResetWithNeuralNetwork(population[currentGenome]);
    }

    public void Death(CarAI carAI, double fitness)
    {
        UpdateBestFitness(fitness);

        if (currentGenome < population.Length - 1)
        {
            population[currentGenome++].Fitness = fitness;
            ResetToCurrentGenome(carAI);
        }
        else
        {
            Repopulate(carAI);
        }
    }

    private void UpdateBestFitness(double fitness)
    {
        bestFitness = (fitness > bestFitness) ? fitness : bestFitness;
    }

    private void Repopulate(CarAI carAI)
    {
        SerializationHelper.SerializeStats(currentGeneration, population, ref _generatedStatsNameFile);
        genesPool.Clear();
        currentGeneration++;
        naturallySelected = 0;

        SortPopulation();
        var newPopulation = SelectBestPopulation();
        Crossover(newPopulation);
        Mutate(newPopulation);
        RandomizePopulation(newPopulation, naturallySelected);
        population = newPopulation;
        currentGenome = 0;

        ResetToCurrentGenome(carAI);
    }

    private void Mutate(NeuralNetwork[] newPopulation)
    {
        for (var i = 0; i < naturallySelected; i++)
        {
            for (var j = 0; j < newPopulation[i].Weights.Count; j++)
            {
                if (Random.Range(0f, 1f) < mutationChance)
                {
                    newPopulation[i].Weights[j] = newPopulation[i].Weights[j].Mutate();
                }
            }
        }
    }

    private void Crossover(NeuralNetwork[] newPopulation)
    {
        for (var i = 0; i < numberToCrossover; i += 2)
        {
            int firstParentIndex = i, secondParentIndex = i + 1;

            if (genesPool.Count >= 1)
            {
                var randomIndex1 = Random.Range(0, genesPool.Count);
                int randomIndex2;
                do
                {
                    randomIndex2 = Random.Range(0, genesPool.Count);
                } while (genesPool[randomIndex1] == genesPool[randomIndex2]);

                firstParentIndex = genesPool[randomIndex1];
                secondParentIndex = genesPool[randomIndex2];
            }

            var children = new[]
            {
                NeuralNetwork.Of(inputsAmount, outputsAmount, activationFunctions),
                NeuralNetwork.Of(inputsAmount, outputsAmount, activationFunctions)
            };

            foreach (var child in children)
            {
                child.Init(inputsAmount, neuronsInHiddenLayerCount, outputsAmount, activationFunctions);
                child.Fitness = 0;
            }

            for (var w = 0; w < children[0].Weights.Count; w++)
            {
                var doCrossover = Random.Range(0f, 1f) < crossoverChance;
                foreach (var child in children)
                {
                    child.Weights[w] = population[doCrossover ? firstParentIndex : secondParentIndex].Weights[w];
                }
            }

            for (var b = 0; b < children[0].Weights.Count; b++)
            {
                var doCrossover = Random.Range(0f, 1f) < crossoverChance;
                foreach (var child in children)
                {
                    child.Biases[b] = population[doCrossover ? firstParentIndex : secondParentIndex].Biases[b];
                }
            }

            foreach (var child in children)
            {
                newPopulation[naturallySelected++] = child;
            }
        }
    }

    private NeuralNetwork[] SelectBestPopulation()
    {
        var newPopulation = new NeuralNetwork[initialPopulation];
        for (var i = 0; i < bestAgentSelection; i++)
        {
            newPopulation[naturallySelected] = population[i].Clone() as NeuralNetwork;
            newPopulation[naturallySelected++].Fitness = 0;

            var f = Mathf.RoundToInt((float) population[i].Fitness * fitnessMultiplier);

            for (var j = 0; j < f; j++)
            {
                genesPool.Add(i);
            }
        }

        for (var i = 0; i < worstAgentSelection; i++)
        {
            var last = population.Length - 1;
            last -= i;

            var f = Mathf.RoundToInt((float) population[last].Fitness * fitnessMultiplier);

            for (var j = 0; j < f; j++)
            {
                genesPool.Add(last);
            }
        }

        return newPopulation;
    }

    private void SortPopulation()
    {
        Array.Sort(population, (n1, n2) => decimal.Compare((decimal) n2.Fitness, (decimal) n1.Fitness));
    }

    public void Configure(Dictionary<string, string> configMap)
    {
        initialPopulation = int.Parse(configMap["initialPopulation"]);
        mutationChance = double.Parse(configMap["mutationChance"]);
        fitnessMultiplier = int.Parse(configMap["fitnessMultiplier"]);
        bestAgentSelection = int.Parse(configMap["bestAgentSelection"]);
        worstAgentSelection = int.Parse(configMap["worstAgentSelection"]);
        numberToCrossover = int.Parse(configMap["numberToCrossover"]);
        crossoverChance = float.Parse(configMap["crossoverChance"]);
    }

    public void Create(NEAT_Manager deserializeNeat)
    {
        _generatedStatsNameFile = "";
        initialPopulation = deserializeNeat.initialPopulation;
        mutationChance = deserializeNeat.mutationChance;
        fitnessMultiplier = deserializeNeat.fitnessMultiplier;
        bestAgentSelection = deserializeNeat.bestAgentSelection;
        worstAgentSelection = deserializeNeat.worstAgentSelection;
        numberToCrossover = deserializeNeat.numberToCrossover;
        crossoverChance = deserializeNeat.crossoverChance;
        currentGeneration = deserializeNeat.currentGeneration;
        currentGenome = deserializeNeat.currentGenome;
        bestFitness = deserializeNeat.bestFitness;
        naturallySelected = deserializeNeat.naturallySelected;
        population = deserializeNeat.population;
        inputsAmount = deserializeNeat.inputsAmount;
        outputsAmount = deserializeNeat.outputsAmount;
        activationFunctions = deserializeNeat.activationFunctions;
        neuronsInHiddenLayerCount = deserializeNeat.neuronsInHiddenLayerCount;
        genesPool = new List<int>();
    }
}
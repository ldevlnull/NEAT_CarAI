using System;
using System.Collections.Generic;
using UnityEngine;
using Random = UnityEngine.Random;

// ReSharper disable once InconsistentNaming
public class NEAT_Manager : MonoBehaviour, IConfigurable
{
    private string _generatedStatsNameFile = "";
    
    [Header("References")] 
    [SerializeField] private CarAI carAI;

    [Header("NEAT Control")] 
    [SerializeField] private int initialPopulation;

    [SerializeField] [Range(0f, 1f)] private double mutationChance;
    [SerializeField] private int fitnessMultiplier = 10;

    [Header("Crossover Control")] 
    [SerializeField] private int bestAgentSelection = 8;
    [SerializeField] private int worstAgentSelection = 3;
    [SerializeField] private int numberToCrossover;
    [SerializeField] [Range(0, 1)] private double crossoverChance = 0.5;

    [Header("Public View")] 
    [SerializeField] private int currentGeneration = 1;
    [SerializeField] private int currentGenome = 1;
    [SerializeField] private double bestFitness;

    private readonly List<int> _genesPool = new List<int>();
    private int _naturallySelected;

    private NeuralNetwork[] _population;

    private void Start()
    {
        GUIHelper.AddToDisplay("Best fitness", () => bestFitness);
        GUIHelper.AddToDisplay("Generation", () => currentGeneration);
        GUIHelper.AddToDisplay("Genome", () => currentGenome);
        
        CreatePopulation();
    }

    private void CreatePopulation()
    {
        _population = new NeuralNetwork[initialPopulation];
        RandomizePopulation(_population, 0);
        ResetToCurrentGenome();
    }

    private void RandomizePopulation(NeuralNetwork[] newPopulation, int startingIndex)
    {
        while (startingIndex < initialPopulation)
        {
            newPopulation[startingIndex] = NeuralNetwork.Of(carAI.InputsAmount, carAI.OutputsAmount, carAI.ActivationsFunctions);
            newPopulation[startingIndex++].Init(carAI.InputsAmount, carAI.neuronsInHiddenLayerCount, carAI.OutputsAmount,
                carAI.ActivationsFunctions);
        }
    }

    private void ResetToCurrentGenome()
    {
        carAI.ResetWithNeuralNetwork(_population[currentGenome]);
    }

    public void Death(double fitness)
    {
        UpdateBestFitness(fitness);

        if (currentGenome < _population.Length - 1)
        {
            _population[currentGenome++].Fitness = fitness;
            ResetToCurrentGenome();
        }
        else
        {
            Repopulate();
        }
    }

    private void UpdateBestFitness(double fitness)
    {
        bestFitness = (fitness > bestFitness) ? fitness : bestFitness;
    }

    private void Repopulate()
    {
        SerializationHelper.SerializeStats(currentGeneration, _population, ref _generatedStatsNameFile);
        _genesPool.Clear();
        currentGeneration++;
        _naturallySelected = 0;

        SortPopulation();
        var newPopulation = SelectBestPopulation();
        Crossover(newPopulation);
        Mutate(newPopulation);
        RandomizePopulation(newPopulation, _naturallySelected);
        _population = newPopulation;
        currentGenome = 0;
        ResetToCurrentGenome();
    }

    private void Mutate(NeuralNetwork[] newPopulation)
    {
        for (var i = 0; i < _naturallySelected; i++)
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

            if (_genesPool.Count >= 1)
            {
                var randomIndex1 = Random.Range(0, _genesPool.Count);
                int randomIndex2;
                do
                {
                    randomIndex2 = Random.Range(0, _genesPool.Count);
                } while (_genesPool[randomIndex1] == _genesPool[randomIndex2]);

                firstParentIndex = _genesPool[randomIndex1];
                secondParentIndex = _genesPool[randomIndex2];
            }

            var children = new[] {
                NeuralNetwork.Of(carAI.InputsAmount, carAI.OutputsAmount, carAI.ActivationsFunctions), 
                NeuralNetwork.Of(carAI.InputsAmount, carAI.OutputsAmount, carAI.ActivationsFunctions)
            };

            foreach (var child in children)
            {
                child.Init(carAI.InputsAmount, carAI.neuronsInHiddenLayerCount, carAI.OutputsAmount, carAI.ActivationsFunctions);
                child.Fitness = 0;
            }

            for (var w = 0; w < children[0].Weights.Count; w++)
            {
                var doCrossover = Random.Range(0f, 1f) < crossoverChance;
                foreach (var child in children)
                {
                    child.Weights[w] = _population[doCrossover ? firstParentIndex : secondParentIndex].Weights[w];
                }
            }
            
            for (var b = 0; b < children[0].Weights.Count; b++)
            {
                var doCrossover = Random.Range(0f, 1f) < crossoverChance;
                foreach (var child in children)
                {
                    child.Biases[b] = _population[doCrossover ? firstParentIndex : secondParentIndex].Biases[b];
                }
            }

            foreach (var child in children)
            {
                newPopulation[_naturallySelected++] = child;
            }
        }
    }

    private NeuralNetwork[] SelectBestPopulation()
    {
        var newPopulation = new NeuralNetwork[initialPopulation];
        for (var i = 0; i < bestAgentSelection; i++)
        {
            newPopulation[_naturallySelected] = _population[i].Clone() as NeuralNetwork;
            newPopulation[_naturallySelected++].Fitness = 0;

            var f = Mathf.RoundToInt((float)_population[i].Fitness * fitnessMultiplier);

            for (var j = 0; j < f; j++)
            {
                _genesPool.Add(i);
            }
        }

        for (var i = 0; i < worstAgentSelection; i++)
        {
            var last = _population.Length - 1;
            last -= i;

            var f = Mathf.RoundToInt((float)_population[last].Fitness * fitnessMultiplier);

            for (var j = 0; j < f; j++)
            {
                _genesPool.Add(last);
            }
        }

        return newPopulation;
    }

    private void SortPopulation()
    {
        Array.Sort(_population, (n1, n2) => decimal.Compare((decimal) n2.Fitness, (decimal) n1.Fitness));
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
}
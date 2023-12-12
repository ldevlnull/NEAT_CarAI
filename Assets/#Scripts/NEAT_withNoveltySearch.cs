using System;
using System.Collections.Generic;
using UnityEngine;
using Random = UnityEngine.Random;

public class NEAT_withNoveltySearch : NEAT_Manager
{

    public List<NeuralNetwork> archive = new List<NeuralNetwork>();
    public double noveltyThreshold = 0.5;
    private double _bestNovelty;
    
    public override void Death(double behaviourScore)
    {
        double novelty = CalculateNovelty(population[currentGenome]);
        
        if (novelty > noveltyThreshold)
        {
            archive.Add(population[currentGenome].Clone() as NeuralNetwork);
        }
        
        UpdateBestNovelty(novelty);

        if (currentGenome < population.Length - 1)
        {
            population[currentGenome++].Novelty = novelty;
            ResetToCurrentGenome();
        }
        else
        {
            Repopulate();
        }
    }

    private double CalculateNovelty(NeuralNetwork genome)
    {
        double sumSquaredDistances = 0.0;

        foreach (var archiveGenome in archive)
        {
            double distance = ComputeEuclideanDistance(genome, archiveGenome);
            sumSquaredDistances += distance * distance;
        }
        double averageSquaredDistance = sumSquaredDistances / archive.Count;
        
        return Math.Sqrt(averageSquaredDistance);
    }
    
    private double ComputeEuclideanDistance(NeuralNetwork genomeA, NeuralNetwork genomeB)
    {
        double squaredSum = 0.0;

        for (int i = 0; i < genomeA.Weights.Count; i++)
        {
            double weightDifference = 0;
            Matrix matrixDif = genomeA.Weights[i] - genomeB.Weights[i];
            for (int row = 0; row < matrixDif.RowsCount; row++)
            {
                for (int col = 0; col < matrixDif.ColsCount; col++)
                {
                    weightDifference += matrixDif[row, col];
                }
            }
            squaredSum += weightDifference * weightDifference;
        }

        return Math.Sqrt(squaredSum);
    }
    
    private void UpdateBestNovelty(double novelty)
    {
        _bestNovelty = Math.Max(novelty, _bestNovelty);
    }
}
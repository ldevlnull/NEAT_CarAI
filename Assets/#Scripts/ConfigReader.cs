using System.Collections.Generic;
using Newtonsoft.Json;
using UnityEngine;

public class ConfigReader : MonoBehaviour
{
    [SerializeField] private TextAsset path;
    [SerializeField] private List<CarAI> carAIs;
    [SerializeField] private NEAT_Manager neatManager;

    public void ReadConfigTree(NEAT_Manager neat)
    {
        if (path == null)
        {
            Debug.Log("no config");
            return;
        }

        Debug.Log("Reading config...");

        var config = JsonConvert.DeserializeObject<Dictionary<string, string>>(path.text);

        neat.Configure(config);
        foreach (var carAI in carAIs)
        {
            carAI.Configure(config);
            neat.PostConfigStart(carAI);
        }

        Debug.Log("Config has been read!");
    }
}
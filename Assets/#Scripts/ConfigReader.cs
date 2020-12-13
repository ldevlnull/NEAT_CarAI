using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using Newtonsoft.Json;
using UnityEngine;
using Object = UnityEngine.Object;

public class ConfigReader : MonoBehaviour
{
    private const string Path = "config.json";
    [SerializeField] private CarAI carAI;
    [SerializeField] private NEAT_Manager neatManager;

    private void Start()
    {
        if (!File.Exists(Path))
            return;
        using (JsonReader file = new JsonTextReader(File.OpenText(Path)))
        {
            Debug.Log("Reading config...");

            var config = new JsonSerializer().Deserialize<Dictionary<string, string>>(file);

            carAI.Configure(config);
            neatManager.Configure(config);

            Debug.Log("Config has been read!");
        }
    }
}
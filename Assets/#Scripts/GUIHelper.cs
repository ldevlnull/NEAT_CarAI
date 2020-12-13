using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

public class GUIHelper : MonoBehaviour
{
    private const int FirstDisplayPositionY = 10;
    private const int DisplayStepY = 30;

    private static GUIStyle _style;

    private static readonly Dictionary<string, Func<double>>
        NameToDataResolver = new Dictionary<string, Func<double>>();

    private void Awake()
    {
        _style = new GUIStyle {fontSize = 26};
    }

    public static void AddToDisplay(string dataName, Func<double> dataResolver)
    {
        if (dataName != null && dataResolver != null)
            NameToDataResolver.Add(dataName, dataResolver);
    }
    
    private void OnGUI()
    {
        var pos = FirstDisplayPositionY;
        foreach (var nameToFunc in NameToDataResolver)
        {
            GUI.Label(new Rect(100, pos, 100, 20), $"{nameToFunc.Key}: {nameToFunc.Value()}", _style);
            pos += DisplayStepY;
        }
    }
}
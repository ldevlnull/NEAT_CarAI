using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using UnityEngine;

public interface IConfigurable
{
    void Configure(Dictionary<string, string> configMap);
}

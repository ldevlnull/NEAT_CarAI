using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using JetBrains.Annotations;
using UnityEngine;
using UnityEngine.Serialization;
using VehiclePhysics;
using Debug = UnityEngine.Debug;

public class CarAI : MonoBehaviour, IConfigurable
{
    private enum SensorType
    {
        Front,
        FrontLeft,
        FrontRight,
        Left,
        Right,
        Back,
        FrontRightCenter,
        FrontLeftCenter,
        
        Speed,
        AxeX,
        AxeY,
        AxeZ
    }

    private NeuralNetwork _network;

    [SerializeField] private KeyCode manualResetButton;

    [Header("Import Neural Network")] 
    private string _jsonFilePath;
    private bool _isNeuralNetworkImported;
    [CanBeNull] public TextAsset jsonFile;

    [Header("Neural Network")] 
    [SerializeField] public int[] neuronsInHiddenLayerCount;
    [SerializeField] private double goneDistanceWeight = 1.25f;
    [SerializeField] private double avgSpeedWeight = 0.8f;
    [SerializeField] private double sensorOffRoadPenalty = 0f;
    [SerializeField] private float  maxSensorReadDistance = 25f; 

    [Header("Sensors")] 
    [SerializeField] private Transform sensorFPosition;
    [SerializeField] private Transform sensorFRPosition;
    [SerializeField] private Transform sensorRPosition;
    [SerializeField] private Transform sensorBPosition;
    [SerializeField] private Transform sensorLPosition;
    [SerializeField] private Transform sensorFLPosition;
    [SerializeField] private Transform sensorFLCPosition;
    [SerializeField] private Transform sensorFRCPosition;

    private static readonly Dictionary<SensorType, Transform> SensorByTypeResolver = new Dictionary<SensorType, Transform>();
    private static readonly Dictionary<SensorType, Func<float>> SensorReadActionByTypeResolver = new Dictionary<SensorType, Func<float>>();

    [Header("Control")] 
    [SerializeField] private bool manualControl;
    [SerializeField] [Range(-1f, 1f)] private double acceleration;
    [SerializeField] [Range(-1f, 1f)] private double steering;
    [SerializeField] [Range( 0f, 1f)] private double handbrake;

    [Header("Fitness")] 
    [SerializeField] private double globalPenalty;
    [SerializeField] private long optimalFitness;
    [SerializeField] private long inefficientFitness;
    [SerializeField] private long efficiencyCheckPeriodS;
    [SerializeField] private long runningTimeLimit;
    [SerializeField] private Transform path;

    private Transform[] _checkpoints;
    
    private int _currentCheckpoint = 0;
    private double _globalFitness;
    private double _cachedFitness;
    private double _runningTime;
    private double _avgSpeed;
    private double _goneDistance;
    private double _fitnessSinceLastCheck = 0;
    private double _distanceTravelled;

    private Vector3 _initPosition;
    private Vector3 _lastPosition;
    private Vector3 _initEulerAngles;
    private Quaternion _initRotation;

    public int inputsAmount;
    public int outputsAmount = 3;

    // acceleration, steering, handbrake
    public Func<double, double>[] ActivationsFunctions { get; } =
    {
        Math.Tanh, 
        Math.Tanh, 
        x => 1 / (1 + Math.Exp(-x))
    };

    internal readonly Action<string, CarAI> CheckSurface = (tagName, carAI) =>
    {
        if (tagName.Equals("Road")) return;
        if (tagName.Equals("Checkpoint")) return;
        carAI.Death();
    };

    private readonly SortedDictionary<SensorType, double> _sensors = new SortedDictionary<SensorType, double>();

    private VPVehicleToolkit _vpVehicleToolkit;
    private VPResetVehicle _vpResetVehicle;
    private NEAT_Manager _neatManager;
    private Rigidbody _rigidbody;

    public void ResetWithNeuralNetwork(NeuralNetwork net)
    {
        _network = net;
        Reset();
    }
    
    private void Awake()
    {
        _neatManager = FindObjectOfType<NEAT_Manager>();
        _isNeuralNetworkImported = jsonFile != null;
        _rigidbody = gameObject.GetComponent<Rigidbody>();
        _vpResetVehicle = gameObject.GetComponent<VPResetVehicle>();
        _vpVehicleToolkit = GetComponent<VPVehicleToolkit>();
        _checkpoints = path.GetComponentsInChildren<Transform>();
        
        Array.Sort(_checkpoints, (t1, t2) => string.Compare(t1.name, t2.name, StringComparison.Ordinal));

        GUIHelper.AddToDisplay("Fitness", () => _globalFitness);
        GUIHelper.AddToDisplay("Time", () => _runningTime);
        GUIHelper.AddToDisplay("Avg speed", () => _avgSpeed);
        GUIHelper.AddToDisplay("Current checkpoint", () => _currentCheckpoint+1);

        SensorByTypeResolver.Add(SensorType.Front, sensorFPosition);
        SensorByTypeResolver.Add(SensorType.Back, sensorBPosition);
        SensorByTypeResolver.Add(SensorType.Right, sensorRPosition);
        SensorByTypeResolver.Add(SensorType.Left, sensorLPosition);
        SensorByTypeResolver.Add(SensorType.FrontLeft, sensorFLPosition);
        SensorByTypeResolver.Add(SensorType.FrontRight, sensorFRPosition);
        SensorByTypeResolver.Add(SensorType.FrontLeftCenter, sensorFLCPosition);
        SensorByTypeResolver.Add(SensorType.FrontRightCenter, sensorFRCPosition);
        
        SensorReadActionByTypeResolver.Add(SensorType.Speed, () => _vpVehicleToolkit.speed / 40);
        SensorReadActionByTypeResolver.Add(SensorType.AxeX, () => NormalizeEuler(transform.eulerAngles.x));
        SensorReadActionByTypeResolver.Add(SensorType.AxeY, () => (transform.eulerAngles.y - 180) / 180);
        SensorReadActionByTypeResolver.Add(SensorType.AxeZ, () => (transform.eulerAngles.z - 180) / 180);

        inputsAmount = SensorByTypeResolver.Count + SensorReadActionByTypeResolver.Count;
        
        var curTransform = transform;
        _initPosition = curTransform.position;
        _initEulerAngles = curTransform.eulerAngles;
        _initRotation = curTransform.rotation;
        _lastPosition = _initPosition;

        if (_isNeuralNetworkImported)
        {
            Destroy(_neatManager);
            _network = SerializationHelper.DeserializeNeuralNetwork(this);
        }
        else
        {
            _network = NeuralNetwork.Of(inputsAmount, outputsAmount, ActivationsFunctions);
        }
    }

    private void Start()
    {
        StartCoroutine(KillIfInefficient());
    }
    
    private IEnumerator KillIfInefficient()
    {
        while (true)
        {
            yield return new WaitForSeconds(efficiencyCheckPeriodS);
            // ReSharper disable once Unity.PerformanceCriticalCodeInvocation
            if (!manualControl && _globalFitness - _fitnessSinceLastCheck < inefficientFitness)
            {
                Death();
                _fitnessSinceLastCheck = 0;
                yield break;
            }
            _fitnessSinceLastCheck = _globalFitness;
        }
    }

    private void FixedUpdate()
    {
        if (!_vpVehicleToolkit.isEngineStarted)
            _vpVehicleToolkit.enabled = !_vpVehicleToolkit.enabled;

        if (_rigidbody.isKinematic)
            _rigidbody.isKinematic = false;

        ReadSensors();

        var neuralNetworkOutput = _network.Run(_sensors.Values.ToArray());

        if (!manualControl)
            UpdateControl(neuralNetworkOutput);

        _runningTime += Time.deltaTime;

        ComputeFitness();
        Checkpoint();
        KillIfReachesTimeLimit();
        
        if (Input.GetKeyDown(manualResetButton))
            Death();
    }
    
    private void ReadSensors()
    {
        var edgeHit = new RaycastHit();
        foreach (SensorType sensorType in Enum.GetValues(typeof(SensorType)))
        {
            if (SensorByTypeResolver.TryGetValue(sensorType, out var sensorPosition))
            {
                var initRotation = sensorPosition.rotation;

                const float step = -0.3f;
                for (var i = 0; i < 900; i++)
                {
                    sensorPosition.Rotate(step, 0, 0);
                    if (!Physics.Raycast(sensorPosition.position, sensorPosition.forward, out var hit)) continue;
                    if (!HitsRoad(hit)) break;
                    edgeHit = hit;
                }
                
                Debug.DrawLine(sensorPosition.position, edgeHit.point, Color.green);
                _sensors[sensorType] = Normalize(Vector3.Distance(transform.position, edgeHit.point));
                sensorPosition.rotation = initRotation;
            }
            else if (SensorReadActionByTypeResolver.TryGetValue(sensorType, out var sensorReadFunction))
            {
                _sensors[sensorType] = sensorReadFunction();
            }
            else
            {
                throw new KeyNotFoundException($"No position or action found for sensor type {sensorType}");
            }
        }
    }
    
    private static bool HitsRoad(RaycastHit hit)
    {
        return hit.collider.gameObject.tag.Contains("Road");
    }
    
    private float Normalize(float value)
    {
        return Mathf.Abs(Mathf.Clamp(value, 0, maxSensorReadDistance)  / maxSensorReadDistance);
    }
    
    private void UpdateControl(IReadOnlyList<double> controlUpdate)
    {
        acceleration = controlUpdate[0];
        steering = controlUpdate[1];
        handbrake = controlUpdate[2];
        Accelerate();
        Steer();
        PutHandbrake();
    }
    
    private void Accelerate()
    {
        if (acceleration < 0)
            _vpVehicleToolkit.SetGear(-1);
        else if (acceleration == 0)
            _vpVehicleToolkit.SetGear(0);
        else if (_vpVehicleToolkit.engagedGear == -1 || _vpVehicleToolkit.engagedGear == 0)
            _vpVehicleToolkit.SetGear(0);
        _vpVehicleToolkit.SetThrottle(Mathf.Abs((float)acceleration));
    }

    private void Steer()
    {
        _vpVehicleToolkit.SetSteering((float)steering);
    }

    private void PutHandbrake()
    {
        _vpVehicleToolkit.SetHandbrake(handbrake >= 0.7 ? 1 : 0);
    }
    
    private void ComputeFitness()
    {
        var destination = _checkpoints[_currentCheckpoint].position;
        var curPosition = transform.position;
        _goneDistance = Vector3.Distance(_lastPosition, destination) - Vector3.Distance(curPosition, destination);
        _avgSpeed = _goneDistance / _runningTime;

        _globalFitness = (_goneDistance * goneDistanceWeight) +
                         (_avgSpeed * avgSpeedWeight)
                         - globalPenalty
                         + _cachedFitness;

        if (!_isNeuralNetworkImported && _globalFitness >= optimalFitness && _globalFitness % optimalFitness < 0.1)
        {
            Debug.Log("Saving net!");
            SerializationHelper.SerializeNeuralNetwork(_network, _globalFitness);
        }
    }
    
    private void Checkpoint()
    {
        if (Vector3.Distance(_checkpoints[_currentCheckpoint].position, transform.position) < 3f)
        {
            if (_currentCheckpoint == _checkpoints.Length - 1)
            {
                SerializationHelper.SerializeNeuralNetwork(_network, _globalFitness);
                return;
            }
            _cachedFitness = _globalFitness;
            _currentCheckpoint++;
            _lastPosition = transform.position;
        }
    }

    private void KillIfReachesTimeLimit()
    {
        if (_runningTime >= runningTimeLimit)
            Death();
    }
    
    private void Update()
    {
        CheckPause();
    }

    private static void CheckPause()
    {
        if (Input.GetKeyDown(KeyCode.P))
        {
            Time.timeScale = Time.timeScale == 0 ? 1 : 0;
        }
    }

    public void Reset()
    {
        StopAllCoroutines();
        _runningTime = 0f;
        _avgSpeed = 0;
        _goneDistance = 0f;
        globalPenalty = 0f;
        _currentCheckpoint = 0;
        _cachedFitness = 0f;
        _fitnessSinceLastCheck = 0f;
        _lastPosition = _initPosition;

        _vpResetVehicle.DoReset();

        var curTransform = transform;
        curTransform.position = _initPosition;
        curTransform.eulerAngles = _initEulerAngles;
        curTransform.rotation = _initRotation;
        _rigidbody.isKinematic = true;
        // ReSharper disable once Unity.PerformanceCriticalCodeInvocation
        StartCoroutine(KillIfInefficient());
    }

    private void OnCollisionStay(Collision other)
    {
        CheckSurface(other.gameObject.tag, this);
    }

    private void Death()
    {
        if (_isNeuralNetworkImported)
            Reset();
        else
            _neatManager.Death(_globalFitness);
    }
    
    private static float NormalizeEuler(float x)
    {
        if (0 <= x && x <= 90)
            return x / 90;

        if (270 <= x && x <= 360)
            return x / -360;

        return 0;
    }

    public void Configure(Dictionary<string, string> configMap)
    {
        _jsonFilePath = configMap["jsonFilePath"];
        goneDistanceWeight = float.Parse(configMap["goneDistanceWeight"]);
        avgSpeedWeight = float.Parse(configMap["avgSpeedWeight"]);
        sensorOffRoadPenalty = float.Parse(configMap["sensorOffRoadPenalty"]);
        maxSensorReadDistance = float.Parse(configMap["maxSensorReadDistance"]);
        optimalFitness = long.Parse(configMap["optimalFitness"]);
        inefficientFitness = int.Parse(configMap["inefficientFitness"]);
        efficiencyCheckPeriodS = int.Parse(configMap["efficiencyCheckPeriodS"]);
        runningTimeLimit = long.Parse(configMap["runningTimeLimit"]);
    }
}
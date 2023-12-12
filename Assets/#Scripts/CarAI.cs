using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using JetBrains.Annotations;
using UnityEngine;
using VehiclePhysics;
using Debug = UnityEngine.Debug;

public class CarAI : MonoBehaviour, IConfigurable
{
    private enum SensorType
    {
        Left,
        Right,

        Speed,

        AxeX,
        AxeZ,
    
        FrontDynamic,
        FrontDynamic1,
        FrontDynamic2,
        FrontDynamic3,
        FrontDynamic4,
        FrontDynamic5,
        FrontDynamic6,
        FrontDynamic7,
        FrontDynamic8,
        FrontDynamic9,
        FrontDynamic10,
        
        CheckpointDirection,
    }

    private readonly SensorType[] _dynamicFrontSensorTypes = Enum.GetValues(typeof(SensorType)).Cast<SensorType>()
        .Where(t => t.ToString().Contains("FrontDynamic")).ToArray();

    private SensorType[] _activeSensorTypes;

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
    [SerializeField] private double passedCheckpointsWeight = 0f;
    [SerializeField] private float  maxSensorReadDistance = 25f; 

    [Header("Sensors")] 
    [SerializeField] private Transform sensorRPosition;
    [SerializeField] private Transform sensorBPosition;
    [SerializeField] private Transform sensorLPosition;
    [SerializeField] private Transform sensorDynamicF;

    private static readonly Dictionary<SensorType, Transform> SensorByTypeResolver =
        new Dictionary<SensorType, Transform>();

    private static readonly Dictionary<SensorType, Func<double>> SensorReadActionByTypeResolver =
        new Dictionary<SensorType, Func<double>>();

    [Header("Control")] 
    [SerializeField] private bool manualControl;
    [SerializeField] [Range(-1f, 1f)] private double acceleration;
    [SerializeField] [Range(-1f, 1f)] private double steering;
    [SerializeField] [Range(0f, 1f)] private double handbrake;

    [Header("Fitness")] 
    [SerializeField] private long optimalFitness;
    [SerializeField] private long inefficientFitness;
    [SerializeField] private long efficiencyCheckPeriodS;
    [SerializeField] private long runningTimeLimit;
    [SerializeField] private Transform path;
    [SerializeField] private bool logInputs;

    private Transform[] _checkpoints;

    private int _currentCheckpoint = 0;
    private double _globalFitness;
    private double _cachedFitness;
    private double _runningTime;
    private double _avgSpeed;
    private double _goneDistance;
    private double _fitnessSinceLastCheck = 0;
    private double _distanceTravelled;
    private List<Double> _globalBehaviour;

    private Vector3 _initPosition;
    private Vector3 _lastPosition;
    private Vector3 _initEulerAngles;
    private Quaternion _initRotation;
    private double _lastSpeed;

    [Header("Front Scanner")] 
    [SerializeField] private int horizontalAngle;
    [SerializeField] private float horizontalStep;
    private int _horizontalRaysCount;

    public int inputsAmount;
    public int outputsAmount = 3;

    // acceleration, steering, handbrake
    public Func<double, double>[] ActivationsFunctions { get; } =
    {
        Math.Tanh,
        Math.Tanh,
        x => 1 / (1 + Math.Exp(-x))
    };

    internal readonly Action<string, CarAI> checkSurface = (tagName, carAI) =>
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
        if (_dynamicFrontSensorTypes.Length < _horizontalRaysCount)
            throw new Exception(
                $"You cannot have more rays than existing types! Types {_dynamicFrontSensorTypes.Length} and rays {_horizontalRaysCount}");

        _horizontalRaysCount = (int) (horizontalAngle * (1 / horizontalStep));
        _neatManager = FindObjectOfType<NEAT_Manager>();
        _isNeuralNetworkImported = jsonFile != null;
        _rigidbody = gameObject.GetComponent<Rigidbody>();
        _vpResetVehicle = gameObject.GetComponent<VPResetVehicle>();
        _vpVehicleToolkit = GetComponent<VPVehicleToolkit>();
        _checkpoints = path.GetComponentsInChildren<Transform>().Where(t => t.tag.Contains("Checkpoint")).ToArray();

        Array.Sort(_checkpoints, (t1, t2) => string.Compare(t1.name, t2.name, StringComparison.Ordinal));
        foreach (var cp in _checkpoints.Skip(3))
        {
            cp.gameObject.SetActive(false);
        }

        GUIHelper.AddToDisplay("Fitness", () => _globalFitness);
        GUIHelper.AddToDisplay("Time", () => _runningTime);
        GUIHelper.AddToDisplay("Avg speed", () => _avgSpeed);
        GUIHelper.AddToDisplay("Current checkpoint", () => _currentCheckpoint+1);
        GUIHelper.AddToDisplay("Timescale", () => Time.timeScale);

        SensorByTypeResolver.Add(SensorType.Right, sensorRPosition);
        SensorByTypeResolver.Add(SensorType.Left, sensorLPosition);
        SensorByTypeResolver.Add(SensorType.FrontDynamic, sensorDynamicF);

        SensorReadActionByTypeResolver.Add(SensorType.Speed, () => _vpVehicleToolkit.speed / 40);
        SensorReadActionByTypeResolver.Add(SensorType.AxeX, () => NormalizeEuler(transform.eulerAngles.x));
        SensorReadActionByTypeResolver.Add(SensorType.AxeZ, () => (transform.eulerAngles.z - 180) / 180);
        SensorReadActionByTypeResolver.Add(SensorType.CheckpointDirection, () =>
        {
            var sensorTransform = SensorByTypeResolver[SensorType.FrontDynamic];
            var frontSensorPosition = sensorTransform.position;
            var nextCheckPointPosition = _checkpoints[
                (_currentCheckpoint < _checkpoints.Length) ? _currentCheckpoint : _currentCheckpoint - 1
            ].position;
        
            var targetDir = nextCheckPointPosition - frontSensorPosition;
            var forward = sensorTransform.forward;
            var angle = Vector3.SignedAngle(forward, targetDir, Vector3.up);
            angle = (angle > 0) ? angle - 65 : angle + 65;
            var normalizedAngle = angle / 180 + 0.5;
        
            return normalizedAngle;
        });
        
        inputsAmount = SensorByTypeResolver.Count + SensorReadActionByTypeResolver.Count + _horizontalRaysCount;

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
        
        _activeSensorTypes = Enum.GetValues(typeof(SensorType))
            .Cast<SensorType>()
            .Where(t => !(_dynamicFrontSensorTypes.Contains(t) && !t.Equals(SensorType.FrontDynamic)))
            .ToArray();
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

        var neuralNetworkOutput = _network.Run(_sensors.Values.ToArray(), logInputs);

        if (!manualControl)
            UpdateControl(neuralNetworkOutput);

        _runningTime += Time.deltaTime;
        
        ComputeFitness();
        ComputeBehaviourScore();
        Checkpoint();
        KillIfReachesTimeLimit();

        if (Input.GetKeyDown(manualResetButton))
            Death();
    }

    private void ReadSensors()
    {
        var edgeHit = new RaycastHit();
        foreach (SensorType sensorType in _activeSensorTypes)
        {
            if (SensorByTypeResolver.TryGetValue(sensorType, out var sensorTransform))
            {
                var sensorPos = sensorTransform.position;
                if (sensorType == SensorType.FrontDynamic)
                {
                    var initRotation = sensorTransform.rotation;

                    for (var i = 0; i <= _horizontalRaysCount; i += 1)
                    {
                        edgeHit = FindRoadEdge(sensorTransform, edgeHit);
                        Debug.DrawLine(sensorPos, edgeHit.point, Color.blue);
                        _sensors[_dynamicFrontSensorTypes[i]] = Normalize(Vector3.Distance(sensorPos, edgeHit.point));
                        sensorTransform.Rotate(0, horizontalStep * i * (i % 2 == 0 ? 1 : -1), 0);
                    }

                    sensorTransform.rotation = initRotation;
                }
                else
                {
                    edgeHit = FindRoadEdge(sensorTransform, edgeHit);
                    if (!edgeHit.point.Equals(Vector3.zero))
                    {
                        Debug.DrawLine(sensorPos, edgeHit.point, Color.green);
                        _sensors[sensorType] = Normalize(Vector3.Distance(sensorPos, edgeHit.point));
                    }

                    edgeHit.point = Vector3.zero;
                }
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

    private static RaycastHit FindRoadEdge(Transform sensorTransform, RaycastHit edgeHit)
    {
        var initRotation = sensorTransform.rotation;
        const float step = -0.2f;
        const float maxAngleX = 90;
        const float maxI = maxAngleX * (-1 / step);
        for (var i = 0; i < maxI; i++)
        {
            if (sensorTransform.rotation.x > maxAngleX)
                break;
            sensorTransform.Rotate(step, 0, 0);
            if (!Physics.Raycast(sensorTransform.position, sensorTransform.forward, out var hit))
                continue;
            if (!HitsRoad(hit))
                break;

            edgeHit = hit;
        }

        sensorTransform.rotation = initRotation;
        return edgeHit;
    }

    private static bool HitsRoad(RaycastHit hit)
    {
        return hit.collider.gameObject.tag.Contains("Road");
    }

    private float Normalize(float value)
    {
        return Mathf.Abs(Mathf.Clamp(value, 0, maxSensorReadDistance) / (maxSensorReadDistance));
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
        _vpVehicleToolkit.SetThrottle(Mathf.Abs((float) acceleration));
    }

    private void Steer()
    {
        _vpVehicleToolkit.SetSteering((float) steering);
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
                         + _currentCheckpoint * passedCheckpointsWeight
                         + _cachedFitness;

        if (!_isNeuralNetworkImported && !manualControl && _globalFitness >= optimalFitness &&
            _globalFitness % optimalFitness < 0.1)
        {
            Debug.Log("Saving net due to optimal fitness!");
            SerializationHelper.SerializeNeuralNetwork(_network, _globalFitness);
        }
    }


    private void ComputeBehaviourScore()
    {
        var destination = _checkpoints[_currentCheckpoint].position;
        var curPosition = transform.position;
        _goneDistance = Vector3.Distance(_lastPosition, destination) - Vector3.Distance(curPosition, destination);
        _avgSpeed = _goneDistance / _runningTime;

        _globalBehaviour = new List<double> {_goneDistance, _avgSpeed};
    }

    private void Checkpoint()
    {
        if (Vector3.Distance(_checkpoints[_currentCheckpoint].position, transform.position) < 3f)
        {
            if (_currentCheckpoint == _checkpoints.Length - 1)
            {
                Debug.Log("Saving net due to last checkpoint pass!");
                SerializationHelper.SerializeNeuralNetwork(_network, _globalFitness);
                return;
            }

            _cachedFitness = _globalFitness;
            _currentCheckpoint++;
            _lastPosition = transform.position;
            if (_currentCheckpoint + 3 < _checkpoints.Length)
            {
                _checkpoints[_currentCheckpoint + 2].gameObject.SetActive(true);
                _checkpoints[_currentCheckpoint - 1].gameObject.SetActive(false);
            }
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

        if (Input.GetKeyDown(KeyCode.O))
        {
            Time.timeScale += 1;
        }

        if (Input.GetKeyDown(KeyCode.I))
        {
            Time.timeScale -= (Time.timeScale > 1.5) ? 1 : 0;
        }
    }

    public void Reset()
    {
        StopAllCoroutines();
        _rigidbody.isKinematic = true;
        _runningTime = 0f;
        _avgSpeed = 0;
        _goneDistance = 0f;
        _currentCheckpoint = 0;
        _cachedFitness = 0f;
        _fitnessSinceLastCheck = 0f;
        _lastPosition = _initPosition;

        _rigidbody.velocity = Vector3.zero;
        _rigidbody.angularVelocity = Vector3.zero;
        _vpResetVehicle.DoReset();

        var curTransform = transform;
        curTransform.position = _initPosition;
        curTransform.eulerAngles = _initEulerAngles;
        curTransform.rotation = _initRotation;

        for (var i = 0; i < _checkpoints.Length; i++)
        {
            _checkpoints[i].gameObject.SetActive(i < 3);
        }

        StartCoroutine(KillIfInefficient());
    }

    private void OnCollisionStay(Collision other)
    {
        checkSurface(other.gameObject.tag, this);
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
        goneDistanceWeight = float.Parse(configMap["goneDistanceWeight"]);
        avgSpeedWeight = float.Parse(configMap["avgSpeedWeight"]);
        passedCheckpointsWeight = float.Parse(configMap["passedCheckpointsWeight"]);
        maxSensorReadDistance = float.Parse(configMap["maxSensorReadDistance"]);
        optimalFitness = long.Parse(configMap["optimalFitness"]);
        inefficientFitness = int.Parse(configMap["inefficientFitness"]);
        efficiencyCheckPeriodS = int.Parse(configMap["efficiencyCheckPeriodS"]);
        runningTimeLimit = long.Parse(configMap["runningTimeLimit"]);
    }
}
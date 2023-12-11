using UnityEngine;

public class CheckSurface : MonoBehaviour
{
    private CarAI _carAI;

    private void Start()
    {
        _carAI = GetComponentInParent<CarAI>();
    }
    
    private void OnTriggerEnter(Collider other)
    {
        _carAI.checkSurface(other.tag, _carAI);
    }
}

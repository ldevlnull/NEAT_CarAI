using UnityEngine;

public class CheckSurface : MonoBehaviour
{

    [SerializeField] private CarAI carAI;

    private void OnTriggerEnter(Collider other)
    {
        carAI.CheckSurface(other.tag, carAI);
    }
}

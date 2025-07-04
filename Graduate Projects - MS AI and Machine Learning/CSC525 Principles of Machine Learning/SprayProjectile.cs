using UnityEngine;

public class SprayProjectile : MonoBehaviour
{
    public float speed = 5f;       // Reduced speed for dodging viability
    public float lifetime = 3f;

    private void Start()
    {
        Destroy(gameObject, lifetime); // Self-destruct after a few seconds
    }

    private void Update()
    {
        transform.Translate(Vector3.forward * speed * Time.deltaTime);
    }

    private void OnTriggerEnter(Collider other)
    {
        if (other.CompareTag("Cat"))
        {
            Debug.Log("Spray hit the cat!");
            var cat = other.GetComponent<CatAgent>();
            if (cat != null)
            {
                cat.AddReward(-0.2f);  // Mild penalty, not terminal
                // Do not end episode
            }

            Destroy(gameObject); // Destroy spray on impact
        }
        else
        {
            Destroy(gameObject); // Destroy spray on any other collision
        }
    }
}

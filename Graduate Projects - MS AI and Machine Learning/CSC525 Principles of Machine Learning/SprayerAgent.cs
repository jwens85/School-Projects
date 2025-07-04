using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;

public class SprayerAgent : Agent
{
    [Header("Sprayer Settings")]
    public Transform rayOrigin;
    public GameObject sprayPrefab;
    public float sprayLength = 10f;
    public float sprayPenalty = -0.01f;
    public Transform catTarget;
    public LayerMask detectionMask;

    [Header("Tracking Settings")]
    public float rotationSpeed = 40f;         // Slower rotation to allow dodging
    public float trackingCooldown = 0.5f;     // Updates direction twice per second
    public float sprayCooldown = 1.5f;        // Cooldown between sprays

    private Vector3 lastTargetDirection = Vector3.forward;
    private float lastUpdateTime = -999f;
    private float lastSprayTime = -999f;

    private void Start()
    {
        if (rayOrigin == null)
        {
            Debug.LogError("RayOrigin is not assigned.");
        }
    }

    public override void OnEpisodeBegin()
    {
        lastTargetDirection = transform.forward;
        lastUpdateTime = Time.time;
        lastSprayTime = Time.time;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        if (catTarget != null && rayOrigin != null)
        {
            Vector3 directionToCat = (catTarget.position - rayOrigin.position).normalized;
            sensor.AddObservation(directionToCat);
        }
        else
        {
            sensor.AddObservation(Vector3.zero);
        }
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        // Timed tracking: Update aim every `trackingCooldown` seconds
        if (catTarget != null && Time.time - lastUpdateTime > trackingCooldown)
        {
            Vector3 toCat = (catTarget.position - transform.position).normalized;
            toCat.y = 0f;
            lastTargetDirection = toCat;
            lastUpdateTime = Time.time;
        }

        // Smooth rotation toward last known Cat direction
        Quaternion targetRotation = Quaternion.LookRotation(lastTargetDirection);
        transform.rotation = Quaternion.RotateTowards(transform.rotation, targetRotation, rotationSpeed * Time.deltaTime);

        // Fire ray to check for Cat hit
        RaycastHit hit;
        if (Physics.Raycast(rayOrigin.position, rayOrigin.forward, out hit, sprayLength, detectionMask))
        {
            if (hit.collider.CompareTag("Cat") && Time.time - lastSprayTime > sprayCooldown)
            {
                Debug.Log($"{gameObject.name} sprayed the cat!");
                AddReward(1f);
                lastSprayTime = Time.time;

                if (sprayPrefab != null)
                {
                    GameObject spray = Instantiate(sprayPrefab, rayOrigin.position, Quaternion.identity);
                    spray.transform.forward = rayOrigin.forward;
                }
            }
            else
            {
                AddReward(sprayPenalty);
            }
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActions = actionsOut.ContinuousActions;
        continuousActions[0] = 0f;
    }
}

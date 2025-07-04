using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;

public class CatAgent : Agent
{
    [SerializeField] private float moveSpeed = 3f;
    [SerializeField] private Transform feederLeft;
    [SerializeField] private Transform feederRight;

    [SerializeField] private Transform spawnCenter;
    [SerializeField] private float spawnRangeX = 3.5f;
    [SerializeField] private float spawnRangeZ = 2f;

    private Rigidbody rb;

    public override void Initialize()
    {
        rb = GetComponent<Rigidbody>();
    }

    public override void OnEpisodeBegin()
    {
        float randX = Random.Range(-spawnRangeX, spawnRangeX);
        float randZ = Random.Range(-spawnRangeZ, spawnRangeZ);

        Vector3 offset = new Vector3(randX, 0f, randZ);
        transform.position = spawnCenter.position + offset;

        rb.linearVelocity = Vector3.zero;
        rb.angularVelocity = Vector3.zero;

        transform.rotation = Quaternion.Euler(0f, Random.Range(0f, 360f), 0f);
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(transform.localPosition);   // 3 floats
        sensor.AddObservation(feederLeft.localPosition);  // 3 floats
        sensor.AddObservation(feederRight.localPosition); // 3 floats
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        float moveX = Mathf.Clamp(actions.ContinuousActions[0], -1f, 1f);
        float moveZ = Mathf.Clamp(actions.ContinuousActions[1], -1f, 1f);

        Vector3 move = new Vector3(moveX, 0f, moveZ) * moveSpeed;
        rb.AddForce(move, ForceMode.VelocityChange);

        // Boundary check to ensure the agent stays within the yard
        Vector3 position = transform.position;
        if (Mathf.Abs(position.x) > 10f || Mathf.Abs(position.z) > 5f)
        {
            Debug.Log("CatAgent is out of bounds! Restarting episode.");
            SetReward(-1f);
            EndEpisode();
            return;
        }

        // Reward shaping and penalty for step
        float distToLeft = Vector3.Distance(transform.localPosition, feederLeft.localPosition);
        float distToRight = Vector3.Distance(transform.localPosition, feederRight.localPosition);
        float minDist = Mathf.Min(distToLeft, distToRight);

        AddReward(-0.0005f * minDist); // closer = better
        AddReward(-0.001f);            // discourage idling
        AddReward(+0.002f);            // survival shaping reward
    }

    private void OnTriggerEnter(Collider other)
    {
        if (other.CompareTag("Feeder"))
        {
            Debug.Log("Feeder reached! Reward granted.");
            SetReward(1.0f);
            EndEpisode();
        }
        else if (other.CompareTag("Spray"))
        {
            SetReward(0f); // Minimal penalty for spray
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActionsOut = actionsOut.ContinuousActions;
        continuousActionsOut[0] = Input.GetAxis("Horizontal");
        continuousActionsOut[1] = Input.GetAxis("Vertical");
    }

    private void OnDrawGizmosSelected()
    {
        if (spawnCenter != null)
        {
            Gizmos.color = Color.yellow;
            Gizmos.DrawWireCube(
                spawnCenter.position,
                new Vector3(spawnRangeX * 2f, 0.1f, spawnRangeZ * 2f)
            );
        }
    }
}

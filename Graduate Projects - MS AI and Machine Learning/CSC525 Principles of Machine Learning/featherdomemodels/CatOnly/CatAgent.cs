using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;

public class CatAgent : Agent
{
    [SerializeField] private float moveSpeed = 3f;
    [SerializeField] private Transform feederLeft;
    [SerializeField] private Transform feederRight;

    private Rigidbody rb;
    private Vector3 startPosition;

    public override void Initialize()
    {
        rb = GetComponent<Rigidbody>();
        startPosition = transform.localPosition;
    }

    public override void OnEpisodeBegin()
    {
        transform.localPosition = startPosition;
        rb.linearVelocity = Vector3.zero;
        rb.angularVelocity = Vector3.zero;
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

        float distToLeft = Vector3.Distance(transform.localPosition, feederLeft.localPosition);
        float distToRight = Vector3.Distance(transform.localPosition, feederRight.localPosition);
        float minDist = Mathf.Min(distToLeft, distToRight);

        AddReward(-0.0005f * minDist); // reward shaping: closer = better
        AddReward(-0.001f);            // small penalty per step
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
            SetReward(0f); // No penalty for now
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActionsOut = actionsOut.ContinuousActions;
        continuousActionsOut[0] = Input.GetAxis("Horizontal");
        continuousActionsOut[1] = Input.GetAxis("Vertical");
    }
}

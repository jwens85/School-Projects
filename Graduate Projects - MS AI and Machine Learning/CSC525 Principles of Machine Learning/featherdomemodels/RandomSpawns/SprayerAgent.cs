using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;

public class SprayerAgent : Agent
{
    [SerializeField] private float moveSpeed = 2f;
    [SerializeField] private float turnSpeed = 150f;

    private Rigidbody rb;
    private Vector3 startingPosition;
    private Quaternion startingRotation;

    public override void Initialize()
    {
        rb = GetComponent<Rigidbody>();
        startingPosition = transform.position;
        startingRotation = transform.rotation;
    }

    public override void OnEpisodeBegin()
    {
        // Reset position and velocity (using linearVelocity per Unity 2023+)
        rb.linearVelocity = Vector3.zero;
        rb.angularVelocity = Vector3.zero;
        transform.position = startingPosition;
        transform.rotation = startingRotation;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // Observe agent's local position and orientation
        sensor.AddObservation(transform.localPosition);
        sensor.AddObservation(transform.forward);
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        float move = actions.ContinuousActions[0];  // Forward/backward
        float turn = actions.ContinuousActions[1];  // Turn left/right

        Vector3 forwardMove = transform.forward * move * moveSpeed * Time.deltaTime;
        Vector3 rotation = Vector3.up * turn * turnSpeed * Time.deltaTime;

        rb.MovePosition(rb.position + forwardMove);
        rb.MoveRotation(rb.rotation * Quaternion.Euler(rotation));

        // Small penalty to encourage purposeful movement
        AddReward(-0.001f);
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActions = actionsOut.ContinuousActions;
        continuousActions[0] = Input.GetAxis("Vertical");
        continuousActions[1] = Input.GetAxis("Horizontal");
    }
}

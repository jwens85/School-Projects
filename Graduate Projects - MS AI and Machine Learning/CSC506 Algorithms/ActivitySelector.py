from operator import attrgetter

# Class to represent an activity with start and finish times
class Activity:
    def __init__(self, start, finish):
        self.start = start   # Start time of the activity
        self.finish = finish # Finish time of the activity

# Function to perform the activity selection using a greedy approach
def activity_selection(activities, activities_size):
    # Sort the activities by their finish times in ascending order
    activities.sort(key=attrgetter('finish'))

    # The list to store the selected activities
    chosen_activities = []

    # Select the first activity
    current_activity = activities[0]
    chosen_activities.append(current_activity)

    # Iterate through the remaining activities
    for i in range(1, activities_size):
        # If the start time of the next activity is greater than or equal to the finish time of the current activity
        if activities[i].start >= current_activity.finish:
            chosen_activities.append(activities[i])
            current_activity = activities[i]  # Update the current activity

    return chosen_activities

# Main program
# Create some activities with their start and finish times
activity_1 = Activity(1, 4)
activity_2 = Activity(3, 5)
activity_3 = Activity(0, 6)
activity_4 = Activity(5, 7)
activity_5 = Activity(8, 9)
activity_6 = Activity(5, 9)
activity_list = [activity_1, activity_2, activity_3, activity_4, activity_5, activity_6]

# Perform the activity selection
chosen_activities = activity_selection(activity_list, len(activity_list))

# Output the selected activities
print("Selected Activities:")
for i, activity in enumerate(chosen_activities, start=1):
    print(f"Activity {i}: Start = {activity.start}, Finish = {activity.finish}")

"""
Explanation:
============

This program implements the **Activity Selection** problem using a greedy approach. The goal of the problem is to select the maximum number of activities that don't overlap, meaning their start and finish times do not conflict.

### Components:
1. **Activity Class:**
   - Each `Activity` object has two properties: `start` and `finish`.
   - The `start` property indicates when the activity begins, and the `finish` property indicates when the activity ends.

2. **Greedy Activity Selection Algorithm:**
   - The function `activity_selection()` implements the greedy algorithm for selecting activities.
   - The first step is to sort all activities based on their finish times (in ascending order). This ensures that the algorithm can select the activity that finishes the earliest, maximizing the number of activities that can be chosen.
   - The first activity is always selected since it has the earliest finish time.
   - The algorithm then iterates through the remaining activities:
     - If an activity starts after or at the same time as the previously selected activity finishes, it is selected.
     - The previously selected activity is updated to the current one.
   - The result is a list of non-overlapping activities that maximize the number of possible selections.

### Example Walkthrough:
- Let's say we have the following activities with their start and finish times:
  - Activity 1: Start = 1, Finish = 4
  - Activity 2: Start = 3, Finish = 5
  - Activity 3: Start = 0, Finish = 6
  - Activity 4: Start = 5, Finish = 7
  - Activity 5: Start = 8, Finish = 9
  - Activity 6: Start = 5, Finish = 9

- After sorting by finish time, the activities are ordered as:
  - Activity 1 (Finish: 4), Activity 2 (Finish: 5), Activity 4 (Finish: 7), Activity 5 (Finish: 9), Activity 6 (Finish: 9), Activity 3 (Finish: 6)

- The algorithm selects Activity 1 (Finish: 4) first, then Activity 4 (Finish: 7), and finally Activity 5 (Finish: 9), because they do not overlap.

### Output:
- The selected activities are:
  - Activity 1: Start = 1, Finish = 4
  - Activity 4: Start = 5, Finish = 7
  - Activity 5: Start = 8, Finish = 9

### Limitations:
- This greedy algorithm assumes that activities are provided with fixed start and finish times, and it maximizes the number of non-overlapping activities. 
- It may not be suitable if you need to maximize other metrics, like total duration of activities, rather than the count.

"""

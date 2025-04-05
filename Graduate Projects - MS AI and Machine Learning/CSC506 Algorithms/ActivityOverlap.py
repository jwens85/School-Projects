# Activity class to represent an activity with a name, start time, and finish time
class Activity:
    def __init__(self, name, initial_start_time, initial_finish_time):
        self.name = name  # Name of the activity
        self.start_time = initial_start_time  # Start time of the activity
        self.finish_time = initial_finish_time  # Finish time of the activity

    # Method to check if two activities conflict with each other
    def conflicts_with(self, other_activity):
        # No conflict if this activity finishes before or when the other activity starts
        if self.finish_time <= other_activity.start_time:
            return False
        # No conflict if the other activity finishes before or when this activity starts
        elif other_activity.finish_time <= self.start_time:
            return False
        # In all other cases, the activities conflict
        else:
            return True

# Main program to test Activity objects and conflict checking
activity_1 = Activity('History museum tour', 9, 10)
activity_2 = Activity('Morning mountain hike', 9, 12)
activity_3 = Activity('Boat tour', 11, 14)

print('History museum tour conflicts with Morning mountain hike:',
      activity_1.conflicts_with(activity_2))  # Expected output: True (they overlap)
print('History museum tour conflicts with Boat tour:',
      activity_1.conflicts_with(activity_3))  # Expected output: False (they don't overlap)
print('Morning mountain hike conflicts with Boat tour:',
      activity_2.conflicts_with(activity_3))  # Expected output: True (they overlap)

"""
Explanation:
============

This program checks if two activities **conflict** (or overlap) in time. 

### Components:
1. **Activity Class:**
   - This class has three attributes:
     - `name`: The name of the activity.
     - `start_time`: The time the activity begins.
     - `finish_time`: The time the activity ends.

2. **Method `conflicts_with()`:**
   - This method is used to check if the current activity conflicts (overlaps) with another activity.
   - The rules for checking conflicts are:
     - **No conflict** if the current activity's finish time is **less than or equal to** the other activity's start time.
     - **No conflict** if the other activity's finish time is **less than or equal to** the current activity's start time.
     - **Conflict exists** in any other case (i.e., when the activities overlap in time).

### Example Walkthrough:
- Activity 1: "History museum tour" (Start = 9, Finish = 10)
- Activity 2: "Morning mountain hike" (Start = 9, Finish = 12)
- Activity 3: "Boat tour" (Start = 11, Finish = 14)

1. **History museum tour vs. Morning mountain hike**:
   - The museum tour starts at 9 and finishes at 10.
   - The hike starts at 9 and finishes at 12.
   - Since the hike overlaps with the museum tour (both start at 9), the method will return **True** (they conflict).

2. **History museum tour vs. Boat tour**:
   - The museum tour ends at 10, and the boat tour starts at 11.
   - Since the museum tour ends before the boat tour starts, there is **no overlap**, and the method will return **False**.

3. **Morning mountain hike vs. Boat tour**:
   - The hike finishes at 12, and the boat tour starts at 11.
   - Since these two activities overlap (the boat tour starts while the hike is still ongoing), the method will return **True** (they conflict).

### Output:
The expected output for this example will be:
- History museum tour conflicts with Morning mountain hike: True
- History museum tour conflicts with Boat tour: False
- Morning mountain hike conflicts with Boat tour: True

### Limitations:
- This program only checks **pairwise conflicts** between two activities at a time.
- It doesn't find a schedule of non-overlapping activities, but it can be useful to check conflicts when adding activities to a schedule.
"""

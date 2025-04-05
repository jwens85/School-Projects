class Actor:
  def __init__(self, name, description):
      self.name = name
      self.description = description

  def __str__(self):
      return f"Actor: {self.name}\nDescription: {self.description}\n"


class UseCase:
  def __init__(self, name, actors, description):
      self.name = name
      self.actors = actors
      self.description = description

  def __str__(self):
      actor_names = ', '.join(actor.name for actor in self.actors)
      return f"Use Case: {self.name}\nActors: {actor_names}\nDescription: {self.description}\n"


# Define actors
citizen = Actor("Citizen", "Reports potholes and tracks repair progress.")
public_works_employee = Actor("Public Works Employee", "Logs reports, manages repairs, and tracks costs.")
repair_crew = Actor("Repair Crew", "Handles the physical repair of potholes.")

# Define use cases
use_cases = [
  UseCase("Report Pothole", [citizen], "Citizens report potholes with details like location, size, and severity."),
  UseCase("Log Pothole", [public_works_employee], "Public Works Employee logs the reported pothole with additional details like district and repair priority."),
  UseCase("Assign Work Order", [public_works_employee], "Public Works Employee assigns a repair crew and resources to fix the pothole."),
  UseCase("Repair Pothole", [repair_crew], "Repair Crew fixes the pothole and logs repair data."),
  UseCase("Track Repair Status", [citizen, public_works_employee], "Citizens and Public Works Employee track the status of reported potholes."),
  UseCase("Log Damage Report", [citizen], "Citizens report any damage caused by potholes."),
  UseCase("Generate Repair Report", [public_works_employee], "Public Works Employee generates reports on the repair status and costs."),
]

# Print actors and use cases
print("Actors:\n")
for actor in [citizen, public_works_employee, repair_crew]:
  print(actor)

print("\nUse Cases:\n")
for use_case in use_cases:
  print(use_case)

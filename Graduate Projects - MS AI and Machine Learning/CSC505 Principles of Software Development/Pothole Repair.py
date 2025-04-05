import random

class RepairCrew:
    def __init__(self, crew_id, crew_size, equipment):
        self.crew_id = crew_id
        self.crew_size = crew_size
        self.equipment = equipment

    def __str__(self):
        return f"Crew ID: {self.crew_id}, Crew Size: {self.crew_size}, Equipment: {self.equipment}"

class PotholeRepair:
    def __init__(self, location, size, hours, crew_size, filler_amount, equipment_cost):
        self.id = random.randint(1000, 9999)  # Unique identifier for the pothole report
        self.location = location
        self.size = size
        self.hours = hours
        self.crew_size = crew_size
        self.filler_amount = filler_amount
        self.equipment_cost = equipment_cost
        self.status = "Not Repaired"
        self.actual_cost = None  # Placeholder for the actual cost

        # Define cost constants
        self.labor_cost_per_hour = 20  # Assuming $20 per hour per crew member
        self.filler_cost_per_unit = 1   # $1 per unit of filler material

    def calculate_cost(self):
        labor_cost = self.hours * self.crew_size * self.labor_cost_per_hour
        filler_cost = self.filler_amount * self.filler_cost_per_unit
        total_cost = labor_cost + filler_cost + self.equipment_cost
        return total_cost

    def __str__(self):
        return (f"Pothole ID: {self.id}\n"
                f"Location: {self.location}, Size: {self.size}\n"
                f"Hours Worked: {self.hours}, Crew Size: {self.crew_size}\n"
                f"Filler Amount: {self.filler_amount}, Equipment Cost: {self.equipment_cost}\n"
                f"Status: {self.status}\n"
                f"Estimated Repair Cost: ${self.calculate_cost():.2f}\n"
                f"Actual Repair Cost: ${self.actual_cost:.2f}\n" if self.actual_cost else "Actual Repair Cost: Not Provided\n")

class WorkOrder:
    def __init__(self, pothole, repair_crew):
        self.pothole = pothole
        self.repair_crew = repair_crew
        self.status = "Not Repaired"

    def assign_repair_crew(self, repair_crew):
        self.repair_crew = repair_crew
        self.status = "Work in Progress"
        self.pothole.status = "Work in Progress"

    def update_status(self, status):
        self.status = status
        self.pothole.status = status

    def __str__(self):
        return (f"Work Order for Pothole ID: {self.pothole.id} at {self.pothole.location}, Size: {self.pothole.size}\n"
                f"Assigned Crew: {self.repair_crew}\n"
                f"Status: {self.status}\n")

# In-memory store to simulate database
pothole_reports = {}

def report_pothole():
    location = input("Enter the location of the pothole: ")
    size = int(input("Enter the size of the pothole (on a scale of 1 to 10): "))
    hours = 0  # Initially, no work has been done
    crew_size = 0  # Initially, no crew assigned
    filler_amount = 0  # Initially, no filler material used
    equipment_cost = 0  # Initially, no equipment cost

    pothole_repair = PotholeRepair(
        location=location,
        size=size,
        hours=hours,
        crew_size=crew_size,
        filler_amount=filler_amount,
        equipment_cost=equipment_cost
    )

    pothole_reports[pothole_repair.id] = pothole_repair
    print(f"\nPothole reported successfully! Your tracking ID is {pothole_repair.id}.\n")

def track_pothole_status():
    pothole_id = int(input("Enter your pothole tracking ID: "))
    if pothole_id in pothole_reports:
        pothole = pothole_reports[pothole_id]
        print(f"\nPothole ID: {pothole.id}")
        print(f"Location: {pothole.location}")
        print(f"Status: {pothole.status}\n")
    else:
        print("Pothole ID not found. Please check the ID and try again.")

def assign_work_order():
    print("\nOpen Citizen Reports:")
    open_reports = [pothole for pothole in pothole_reports.values() if pothole.status == "Not Repaired"]

    if not open_reports:
        print("No open reports available.\n")
        return

    # Display the list of open pothole reports with their IDs and locations
    for pothole in open_reports:
        print(f"Pothole ID: {pothole.id}, Location: {pothole.location}")

    pothole_id = int(input("\nPlease select a pothole ID to assign: "))

    if pothole_id in pothole_reports and pothole_reports[pothole_id].status == "Not Repaired":
        pothole_repair = pothole_reports[pothole_id]

        # Proceed with inputting work order details
        hours = float(input("Enter the estimated hours of work: "))
        crew_size = int(input("Enter the number of people on the repair crew: "))
        filler_amount = float(input("Enter the amount of filler material needed (in units): "))
        equipment_cost = float(input("Enter the estimated cost of equipment usage: "))

        # Update the pothole report with these new details
        pothole_repair.hours = hours
        pothole_repair.crew_size = crew_size
        pothole_repair.filler_amount = filler_amount
        pothole_repair.equipment_cost = equipment_cost

        # Calculate and display the estimated cost based on the updated pothole data
        estimated_cost = pothole_repair.calculate_cost()
        print(f"\nEstimated Repair Cost for Pothole ID {pothole_id}: ${estimated_cost:.2f}")

        repair_crew = RepairCrew(crew_id="Crew123", crew_size=crew_size, equipment="Truck, Shovel, Asphalt")

        work_order = WorkOrder(pothole_repair, repair_crew)
        work_order.assign_repair_crew(repair_crew)

        print("\nWork Order Details:")
        print(work_order)
    else:
        print("Pothole ID not found or already assigned. Please check the ID and try again.")

def view_completed_work():
    pothole_id = int(input("Enter the pothole ID to view completed work: "))
    if pothole_id in pothole_reports and pothole_reports[pothole_id].status in ["Repaired", "Rework Complete"]:
        pothole = pothole_reports[pothole_id]
        print(f"\nPothole ID: {pothole.id}")
        print(f"Location: {pothole.location}")
        print(f"Estimated Repair Cost: ${pothole.calculate_cost():.2f}")
        actual_cost = pothole.actual_cost if pothole.actual_cost is not None else "Not Provided"
        print(f"Actual Repair Cost: ${actual_cost}")
    else:
        print("Pothole ID not found or the work is not completed.")

def repair_crew_menu():
    while True:
        print("\nPothole Filling Team Menu")
        print("1. View Assigned Work Orders")
        print("2. Update Work Order Status")
        print("3. Exit")
        choice = input("Select an option (1-3): ")

        if choice == "1":
            print("\nAssigned Work Orders:\n")
            for pothole_id, pothole in pothole_reports.items():
                if pothole.status == "Work in Progress":
                    print(f"Work Order for Pothole ID: {pothole.id} at {pothole.location}")
                    print(f"Size: {pothole.size}, Status: {pothole.status}\n")
        elif choice == "2":
            pothole_id = int(input("Enter the pothole ID to update status: "))
            if pothole_id in pothole_reports:
                print("\nUpdate Status:")
                print("1. Work Not Started")
                print("2. Work in Progress")
                print("3. Work Complete")
                print("4. Rework Complete")
                status_choice = input("Select the new status (1-4): ")

                if status_choice == "1":
                    new_status = "Not Repaired"
                elif status_choice == "2":
                    new_status = "Work in Progress"
                elif status_choice == "3":
                    new_status = "Repaired"
                elif status_choice == "4":
                    new_status = "Rework Complete"
                else:
                    print("Invalid status choice. Please select a valid option.")
                    continue

                pothole_reports[pothole_id].status = new_status

                if new_status in ["Repaired", "Rework Complete"]:
                    # Input the actual cost after repair
                    actual_cost = float(input("Enter the actual cost of the repair: "))
                    pothole_reports[pothole_id].actual_cost = actual_cost

                print("Status and cost updated successfully.")
            else:
                print("Pothole ID not found. Please check the ID and try again.")
        elif choice == "3":
            print("Exiting the Pothole Filling Team menu.")
            break
        else:
            print("Invalid option. Please select a valid choice.")

def citizen_menu():
    while True:
        print("\nCitizen Menu")
        print("1. Report a Pothole")
        print("2. Track Pothole Status")
        print("3. Exit")
        choice = input("Select an option (1-3): ")

        if choice == "1":
            report_pothole()
        elif choice == "2":
            track_pothole_status()
        elif choice == "3":
            print("Exiting the Citizen menu.")
            break
        else:
            print("Invalid option. Please select a valid choice.")

def public_works_menu():
    while True:
        print("\nPublic Works Employee Menu")
        print("1. Assign Work Order")
        print("2. View Completed Work")
        print("3. Exit")
        choice = input("Select an option (1-3): ")

        if choice == "1":
            assign_work_order()
        elif choice == "2":
            view_completed_work()
        elif choice == "3":
            print("Exiting the Public Works Employee menu.")
            break
        else:
            print("Invalid option. Please select a valid choice.")

def main_menu():
    while True:
        print("\nWelcome to the Pothole Tracking and Repair System")
        print("Are you a:")
        print("1. Citizen")
        print("2. Public Works Employee")
        print("3. Pothole Filling Team")
        print("4. Exit")
        choice = input("Select your role (1-4): ")

        if choice == "1":
            citizen_menu()
        elif choice == "2":
            public_works_menu()
        elif choice == "3":
            repair_crew_menu()
        elif choice == "4":
            print("Exiting the Pothole Tracking and Repair System.")
            break
        else:
            print("Invalid option. Please select a valid choice.")

# Run the main menu
main_menu()

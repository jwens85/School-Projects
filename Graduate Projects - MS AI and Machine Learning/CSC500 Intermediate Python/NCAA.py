# def tournament():
#   # Set containing all 64 team names
#   all_teams = {
#       "Alabama", "Houston", "Kansas", "Purdue", "UCLA", "Texas", "Arizona", "Marquette",
#       "Baylor", "Gonzaga", "Kansas State", "Xavier", "UConn", "Tennessee", "Indiana", "Virginia",
#       "San Diego State", "Duke", "Saint Mary’s (CA)", "Miami (FL)", "Iowa State", "Creighton",
#       "Kentucky", "TCU", "Texas A&M", "Michigan State", "Missouri", "Northwestern", "Memphis",
#       "Arkansas", "Maryland", "Iowa", "Florida Atlantic", "West Virginia", "Auburn", "Illinois",
#       "Boise State", "Penn State", "USC", "Utah State", "North Carolina State", "Providence",
#       "Mississippi State", "Pittsburgh", "Arizona State", "Nevada", "College of Charleston", 
#       "Oral Roberts", "Drake", "VCU", "Kent State", "Iona", "Furman", "Louisiana", "Kennesaw State",
#       "UC Santa Barbara", "Grand Canyon", "Montana State", "Vermont", "Colgate", "Princeton",
#       "UNC Asheville", "Northern Kentucky", "Howard"
#   }

#   # Define the teams for each region
#   east_region = ["Alabama", "Baylor", "UConn", "Duke", "Florida Atlantic", "Gonzaga", "Houston", "Indiana",
#                  "Kansas", "Louisiana", "Marquette", "Memphis", "North Carolina State", "Oral Roberts",
#                  "Purdue", "San Diego State"]

#   west_region = ["Kansas State", "Xavier", "Tennessee", "Virginia", "Saint Mary’s (CA)", "Miami (FL)", "Iowa State",
#                  "Creighton", "Kentucky", "TCU", "Texas A&M", "Michigan State", "Missouri", "Northwestern",
#                  "Arkansas", "Maryland"]

#   south_region = ["Iowa", "West Virginia", "Auburn", "Illinois", "Boise State", "Penn State", "USC", "Utah State",
#                   "Providence", "Mississippi State", "Pittsburgh", "Arizona State", "Nevada", "College of Charleston",
#                   "Oral Roberts", "Drake"]

#   midwest_region = ["VCU", "Kent State", "Iona", "Furman", "Louisiana", "Kennesaw State", "UC Santa Barbara",
#                     "Grand Canyon", "Montana State", "Vermont", "Colgate", "Princeton", "UNC Asheville",
#                     "Northern Kentucky", "Howard", "UCLA"]

#   print("East Region Teams:")
#   for team in east_region:
#       print(team)

#   print("\nWest Region Teams:")
#   for team in west_region:
#       print(team)

#   print("\nSouth Region Teams:")
#   for team in south_region:
#       print(team)

#   print("\nMidwest Region Teams:")
#   for team in midwest_region:
#       print(team)
#   print("There were", len(all_teams), "teams starting the 2023 NCAA Men's Basketball Tournament")

# east_region_rankings = {1: "Alabama", 2: "Baylor", 3: "UConn", 4: "Duke", 5: "Florida Atlantic", 6: "Gonzaga",
#   7: "Houston", 8: "Indiana", 9: "Kansas", 10: "Louisiana", 11: "Marquette", 12: "Memphis",
#   13: "North Carolina State", 14: "Oral Roberts", 15: "Purdue", 16: "San Diego State"}

# west_region_rankings = {1: "Kansas State", 2: "Xavier", 3: "Tennessee", 4: "Virginia", 5: "Saint Mary’s (CA)",
#   6: "Miami (FL)", 7: "Iowa State", 8: "Creighton", 9: "Kentucky", 10: "TCU",
#   11: "Texas A&M", 12: "Michigan State", 13: "Missouri", 14: "Northwestern",
#   15: "Arkansas", 16: "Maryland"}

# south_region_rankings = {1: "Iowa", 2: "West Virginia", 3: "Auburn", 4: "Illinois", 5: "Boise State",
#    6: "Penn State", 7: "USC", 8: "Utah State", 9: "Providence", 10: "Mississippi State",
#    11: "Pittsburgh", 12: "Arizona State", 13: "Nevada", 14: "College of Charleston",
#    15: "Oral Roberts", 16: "Drake"}

# midwest_region_rankings = {1: "VCU", 2: "Kent State", 3: "Iona", 4: "Furman", 5: "Louisiana", 6: "Kennesaw State",
#      7: "UC Santa Barbara", 8: "Grand Canyon", 9: "Montana State", 10: "Vermont",
#      11: "Colgate", 12: "Princeton", 13: "UNC Asheville", 14: "Northern Kentucky",
#      15: "Howard", 16: "UCLA"}


# def print_region(region_name, region_rankings):
#     print(f"{region_name} Region Teams:")
#     for rank, team in region_rankings.items():
#         print(f"{rank}: {team}")
#     print()
# print_region("East", east_region_rankings)
# print_region("West", west_region_rankings)
# print_region("South", south_region_rankings)
# print_region("Midwest", midwest_region_rankings)

#Lists and Dictionaries were so Janurary 24, let's get with the times and use what we've learned in week 3

def print_teams(teams):
  for region, region_teams in teams.items():
      print(f"{region} Region Teams:")
      for team in region_teams:
          print(f"Rank {team['rank']}: {team['name']}")
      print()  # Add a blank line for better readability between regions

# Initialize the tournament data structure with teams, regions, and rankings
teams = {
  "East": [
      {"rank": 1, "name": "Alabama"}, {"rank": 2, "name": "Baylor"}, {"rank": 3, "name": "UConn"},
      {"rank": 4, "name": "Duke"}, {"rank": 5, "name": "Florida Atlantic"}, {"rank": 6, "name": "Gonzaga"},
      {"rank": 7, "name": "Houston"}, {"rank": 8, "name": "Indiana"}, {"rank": 9, "name": "Kansas"},
      {"rank": 10, "name": "Louisiana"}, {"rank": 11, "name": "Marquette"}, {"rank": 12, "name": "Memphis"},
      {"rank": 13, "name": "North Carolina State"}, {"rank": 14, "name": "Purdue"}, {"rank": 15, "name": "San Diego State"},
      {"rank": 16, "name": "Oral Roberts"}
  ],
  "West": [
      {"rank": 1, "name": "Kansas State"}, {"rank": 2, "name": "Xavier"}, {"rank": 3, "name": "Tennessee"},
      {"rank": 4, "name": "Virginia"}, {"rank": 5, "name": "Saint Mary’s (CA)"}, {"rank": 6, "name": "Miami (FL)"},
      {"rank": 7, "name": "Iowa State"}, {"rank": 8, "name": "Creighton"}, {"rank": 9, "name": "Kentucky"},
      {"rank": 10, "name": "TCU"}, {"rank": 11, "name": "Texas A&M"}, {"rank": 12, "name": "Michigan State"},
      {"rank": 13, "name": "Missouri"}, {"rank": 14, "name": "Northwestern"}, {"rank": 15, "name": "Arkansas"},
      {"rank": 16, "name": "Maryland"}
  ],
  "South": [
      {"rank": 1, "name": "Iowa"}, {"rank": 2, "name": "West Virginia"}, {"rank": 3, "name": "Auburn"},
      {"rank": 4, "name": "Illinois"}, {"rank": 5, "name": "Boise State"}, {"rank": 6, "name": "Penn State"},
      {"rank": 7, "name": "USC"}, {"rank": 8, "name": "Utah State"}, {"rank": 9, "name": "Providence"},
      {"rank": 10, "name": "Mississippi State"}, {"rank": 11, "name": "Pittsburgh"}, {"rank": 12, "name": "Arizona State"},
      {"rank": 13, "name": "Nevada"}, {"rank": 14, "name": "College of Charleston"}, {"rank": 15, "name": "Drake"},
      {"rank": 16, "name": "VCU"}
  ],
  "Midwest": [
      {"rank": 1, "name": "UCLA"}, {"rank": 2, "name": "Kent State"}, {"rank": 3, "name": "Iona"},
      {"rank": 4, "name": "Furman"}, {"rank": 5, "name": "Louisiana"}, {"rank": 6, "name": "Kennesaw State"},
      {"rank": 7, "name": "UC Santa Barbara"}, {"rank": 8, "name": "Grand Canyon"}, {"rank": 9, "name": "Montana State"},
      {"rank": 10, "name": "Vermont"}, {"rank": 11, "name": "Colgate"}, {"rank": 12, "name": "Princeton"},
      {"rank": 13, "name": "UNC Asheville"}, {"rank": 14, "name": "Northern Kentucky"}, {"rank": 15, "

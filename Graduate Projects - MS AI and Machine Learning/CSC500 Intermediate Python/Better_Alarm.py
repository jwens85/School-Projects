def alarm_clock():
    def input_with_validation(prompt, min_value, max_value):
        while True:
            try:
                value = int(input(prompt))
                if min_value <= value <= max_value:
                    return value
                else:
                    print(f"Please enter a value between {min_value} and {max_value}.")
            except ValueError:
                print("Invalid input. Please enter a number.")

    current_hour = input_with_validation("Enter the current hour in a 24h format (0-23): ", 0, 23)
    current_minute = input_with_validation("Enter the current minute (0-59): ", 0, 59)
    wait_hours = input_with_validation("Enter the number of hours to wait: ", 0, 23)
    wait_minutes = input_with_validation("Enter the number of minutes to wait: ", 0, 59)
    alarm_label = input("Enter an alarm label (optional): ").strip() or "Alarm"

    total_current_minutes = current_hour * 60 + current_minute
    total_wait_minutes = wait_hours * 60 + wait_minutes
    total_alarm_minutes = total_current_minutes + total_wait_minutes

    new_hour, new_minute = divmod(total_alarm_minutes, 60)
    new_hour %= 24

    if new_hour == 0 and new_minute == 0:
        time_str = "12 Midnight"
    elif new_hour == 12 and new_minute == 0:
        time_str = "12 Noon"
    else:
        phase = "AM" if new_hour < 12 or new_hour == 24 else "PM"
        new_hour = new_hour - 12 if new_hour > 12 else new_hour
        new_hour = 12 if new_hour == 0 or new_hour == 24 else new_hour
        time_str = f"{new_hour:02d}:{new_minute:02d} {phase}"

    hour_str = "hour" if wait_hours == 1 else "hours"
    minute_str = "minute" if wait_minutes == 1 else "minutes"
    print(f"{alarm_label}: It will be {time_str} when the alarm rings, after {wait_hours} {hour_str} and {wait_minutes} {minute_str}.")

alarm_clock()
def read_file_to_dict(filename):
    """Reads a text file and returns a dictionary of student names and grades."""
    student_dict = {}
    try:
        with open(filename, "r") as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) == 2:
                    name, grade = parts
                    try:
                        student_dict[name] = float(grade)
                    except ValueError:
                        print(f"Skipping invalid grade for {name}.")
        return student_dict
    except FileNotFoundError:
        print(f"File '{filename}' not found. Starting with an empty record.")
        return {}


def add_student(student_dict, name, grade):
    """Adds a new student and grade to the dictionary."""
    try:
        grade = float(grade)
        student_dict[name] = grade
        print(f"Added {name} with grade {grade}.")
    except ValueError:
        print("Invalid grade. Please enter a numeric value.")


def calculate_average(student_dict):
    """Calculates and returns the average grade."""
    if not student_dict:
        print("No student data available.")
        return 0
    average = sum(student_dict.values()) / len(student_dict)
    print(f"Average grade: {average:.2f}")
    return average


def print_above_average_students(student_dict, average):
    """Prints students who scored above the average grade."""
    print("\nStudents scoring above average:")
    found = False
    for name, grade in student_dict.items():
        if grade > average:
            print(f"{name}: {grade}")
            found = True
    if not found:
        print("No students scored above average.")


# Main program
if __name__ == "__main__":
    filename = "students.txt"
    students = read_file_to_dict(filename)

    while True:
        print("\nOptions:")
        print("1. Add new student data")
        print("2. Calculate average grade")
        print("3. Print students who scored above average")
        print("4. Exit")

        choice = input("Enter your choice: ")

        if choice == "1":
            name = input("Enter student name: ")
            grade = input("Enter student grade: ")
            add_student(students, name, grade)

        elif choice == "2":
            average = calculate_average(students)

        elif choice == "3":
            if not students:
                print("No data to analyze.")
            else:
                average = calculate_average(students)
                print_above_average_students(students, average)

        elif choice == "4":
            print("Exiting program. Goodbye!")
            break

        else:
            print("Invalid choice. Please try again.")

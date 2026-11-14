
import random
import json
import itertools

task_counter = itertools.count(101)

task_types = {
    "1": "matrix_multiplication",
    "2": "sorting",
    "3": "numerical_computation",
    "4": "data_preprocessing"
}

default_sizes = {
    "matrix_multiplication": 100,
    "sorting": 10000,
    "numerical_computation": 10000,
    "data_preprocessing": 10000
}


def create_task(operation, size=None):

    if operation not in default_sizes:
        raise ValueError("Invalid task operation")

    if size is None:
        size = default_sizes[operation]

    limits = {
        "matrix_multiplication": 100,
        "sorting": 1000000,
        "numerical_computation": 1000000,
        "data_preprocessing": 1000000
    }

    if not isinstance(size, int) or size < 1 or size > limits[operation]:
        raise ValueError("Invalid task size")

    task = {
        "task_id": next(task_counter),
        "operation": operation,
        "size": size
    }

    return task


def generate_random_task():

    operation = random.choice(list(default_sizes.keys()))

    size = random.choice([
        default_sizes[operation],
        min(default_sizes[operation] * 2, 1000000),
        min(default_sizes[operation] * 5, 1000000)
    ])

    if operation == "matrix_multiplication":
        size = min(size, 100)

    return create_task(operation, size)


def generate_batch(count):

    tasks = []

    for i in range(count):
        task = generate_random_task()
        tasks.append(task)

    return tasks


def display_task(task):

    print("\nGenerated Task:")
    print(json.dumps(task, indent=4))


def main():

    while True:

        print("\n===== TASK GENERATOR =====")
        print("1. Matrix Multiplication")
        print("2. Array Sorting")
        print("3. Numerical Computation")
        print("4. Data Preprocessing")
        print("5. Generate Random Task")
        print("6. Generate Multiple Random Tasks")
        print("7. Exit")

        choice = input("\nEnter your choice: ")

        if choice in task_types:

            operation = task_types[choice]

            size = input("Enter task size (press Enter for default): ")

            if size == "":
                task = create_task(operation)
            else:
                task = create_task(operation, int(size))

            display_task(task)

        elif choice == "5":

            task = generate_random_task()
            display_task(task)

        elif choice == "6":

            count = int(input("How many tasks to generate? "))

            if count < 1 or count > 1000:
                print("Enter a count between 1 and 1000")
                continue

            tasks = generate_batch(count)

            print("\nGenerated Tasks:")
            print(json.dumps(tasks, indent=4))

        elif choice == "7":

            print("Task Generator Stopped")
            break

        else:

            print("Invalid choice")


if __name__ == "__main__":
    main()

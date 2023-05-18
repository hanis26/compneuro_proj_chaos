import numpy as np
import matplotlib.pyplot as plt


def update_activities(activities, weights):
    return np.tanh(np.dot(weights, activities))


def apply_perturbation(activities, perturbation_set, epsilon):
    activities[perturbation_set] += epsilon
    return activities


def calculate_distance(activities1, activities2):
    return np.linalg.norm(activities1 - activities2)


def simulate_network(
    num_neurons, perturbation_set, epsilon, num_steps, transition_point
):
    # Initialize activities randomly
    activities = np.random.uniform(-1, 1, num_neurons)

    # Initialize weights randomly near the transition point
    weights = np.random.normal(0, 1 / np.sqrt(num_neurons), (num_neurons, num_neurons))

    # Apply updates until reaching steady state
    for _ in range(num_steps):
        activities = update_activities(activities, weights)

    activities_perturbed = apply_perturbation(
        np.copy(activities), perturbation_set, epsilon
    )

    # Calculate distances over time
    distances = []
    for _ in range(num_steps):
        activities = update_activities(activities, weights)
        activities_perturbed = update_activities(activities_perturbed, weights)
        distance = calculate_distance(activities_perturbed, activities)
        distances.append(distance)

    return distances, activities, activities_perturbed


# Set the parameters
num_neurons = 5000  # Number of neurons in the network
perturbation_set = [500,501,502,503,504]  # Indices of neurons to perturb
epsilon = 0.1  # Magnitude of perturbation
num_steps = 10000  # Number of time steps to run the simulation
transition_point = (
    0.2  # Controls the network's proximity to the chaotic transition point
)
x = np.arange(0, num_neurons, 1)

distances, activities, perturbed_activities = simulate_network(
    num_neurons, perturbation_set, epsilon, num_steps, transition_point
)

# Calculate the Lyapunov exponent
time = np.arange(num_steps)
fit = np.polyfit(time, np.log(distances), 1)
lyapunov_exponent = -fit[0]

# Print the Lyapunov exponent
print("Lyapunov exponent:", lyapunov_exponent)


plt.figure()
plt.figure()
plt.plot(x,activities,label='unperturbed')
plt.xlabel("Neuron ID")
plt.ylabel("Activity")
plt.legend()
plt.title("Unperturbed Neurons")
plt.show()

plt.plot(x,perturbed_activities,label='perturbed')
plt.xlabel("Neuron ID")
plt.ylabel("Activity")
plt.title("Perturbed Neurons")
plt.legend()
plt.show()


plt.plot(time, distances)
plt.xlabel("Time")
plt.ylabel("Distance (d)")
plt.title("Distance between Perturbed and Unperturbed States")
plt.show()
import matplotlib.pyplot as plt

label_percentages_x = [0.05, 0.1, 0.2, 0.5, 1]
test_accuracies_algo = [0.966, 0.898, 0.864, 0.93, 0.912]
test_accuracies_random = [0.966,0.956,0.960,0.974,0.972]
test_accuracies_closest = [0.756, 0.954, 0.948, 0.926, 0.876]
test_accuracies_outskirts = [0.928, 0.952, 0.938, 0.856, 0.876]
test_accuracies_outskirts_and_closest = [0.942, 0.934, 0.968, 0.95, 0.876]
test_accuracies_middle = [0.958, 0.86, 0.866, 0.924, 0.958]


# Plotting the data points
plt.plot(label_percentages_x, test_accuracies_algo, label='our algorithm', color='blue',marker = '*')
plt.plot(label_percentages_x, test_accuracies_outskirts_and_closest, label='furthest and closest', color='red',marker = '>')
plt.plot(label_percentages_x, test_accuracies_outskirts, label='furthest to center', color='black',marker = '+')
plt.plot(label_percentages_x, test_accuracies_closest, label='closest to center', color='green',marker = '<')
plt.plot(label_percentages_x, test_accuracies_middle, label='medium distance to center', color='pink',marker = 'p')
plt.plot(label_percentages_x, test_accuracies_random, label='random algorithm', color='orange',marker = '.')
plt.axvline(x=0.05,linestyle='--',linewidth=2,color='#A9A9A9')
plt.axvline(x=0.1,linestyle='--',linewidth=2,color='#A9A9A9')
plt.axvline(x=0.2,linestyle='--',linewidth=2,color='#A9A9A9')
plt.axvline(x=0.5,linestyle='--',linewidth=2,color='#A9A9A9')
plt.axvline(x=1,linestyle='--',linewidth=2,color='#A9A9A9')
plt.xticks([0.05, 0.1, 0.2, 0.5, 1], ['5%', '10%', '20%', '50%', '100%'])
plt.ylim(0, 1)


# Adding labels and title
plt.xlabel('Label Percentage')
plt.ylabel('Test Accuracy')
plt.title('Accuracy for each algorithm on synthetic data (k=10)')
plt.legend()

# Display the plot
plt.show()
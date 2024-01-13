import matplotlib.pyplot as plt

label_percentages_x = [0.05, 0.1, 0.2, 0.5, 1]
test_accuracies_algo = [0.608, 0.854, 0.892, 0.844, 0.916]
test_accuracies_random = [0.870,0.892,0.906,0.918,0.932]
test_accuracies_closest = [0.492, 0.606, 0.734, 0.886, 0.824]
test_accuracies_outskirts = [0.5, 0.638, 0.816, 0.81, 0.78]
test_accuracies_outskirts_and_closest = [0.438, 0.754, 0.64, 0.708, 0.78]
test_accuracies_middle = [0.484, 0.666, 0.83, 0.602, 0.916]


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
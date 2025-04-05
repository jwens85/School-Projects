import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(12, 12))

# Parameters for the inward to outward spiral wedges
labels = ['Communication', 'Planning', 'Modeling', 'Construction', 'Deployment', 'Release']
colors = ['skyblue', 'lightgreen', 'magenta', 'orange', 'red', 'purple']
theta = np.linspace(0, 2 * np.pi, len(labels) + 1)
outer_radius = 3
radius_step = outer_radius / len(labels)

# Draw spiral wedges from the center outwards
for i, label in enumerate(labels):
    # For an inside-out spiral, we start from the center
    inner_radius = i * radius_step
    outer_radius = (i + 1) * radius_step
    wedge = patches.Wedge(center=(0, 0), r=outer_radius, theta1=np.degrees(theta[i]),
                          theta2=np.degrees(theta[i + 1]), width=radius_step, facecolor=colors[i], edgecolor='black')
    ax.add_patch(wedge)

    # Label positioning at the middle of the wedge radius
    label_radius = inner_radius + radius_step / 2
    label_theta = (theta[i] + theta[i + 1]) / 2
    label_x = label_radius * np.cos(label_theta)
    label_y = label_radius * np.sin(label_theta)
    ax.text(label_x, label_y, label, ha='center', va='center', color='white', weight='bold')

# Draw arrows
for i in range(len(labels)):
    start_radius = (i + 1) * radius_step - radius_step / 2
    start_theta = theta[i + 1]
    end_radius = (i + 1.5) * radius_step
    end_theta = theta[i + 1] + np.pi / 8

    # Start point
    start_x = start_radius * np.cos(start_theta)
    start_y = start_radius * np.sin(start_theta)

    # End point
    end_x = end_radius * np.cos(end_theta)
    end_y = end_radius * np.sin(end_theta)

    # Draw arrow
    ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                arrowprops=dict(arrowstyle="->", color='black', lw=2))

ax.set_xlim(-outer_radius, outer_radius)
ax.set_ylim(-outer_radius, outer_radius)
ax.set_aspect('equal')
ax.axis('off')
plt.title('Spiral Model Development Phases', fontsize=16, fontweight='bold')
plt.show()

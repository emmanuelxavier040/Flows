# Reimport necessary libraries
import matplotlib.pyplot as plt

# Define the data for original and replicated vectors
total_boxes = 4  # There are always 4 boxes per vector
colors = ['#FF6666', '#D3D3D3']  # Red for replicated, Gray for non-replicated

# Setup the plot with 16 rows and 5 columns (to plot 16 rows of 5 rectangles)
fig, axs = plt.subplots(nrows=16, ncols=5, figsize=(12, 24))  # 16 rows and 5 columns
fig.suptitle('Column Replication Across Multiple Rows', fontsize=16, y=1.05)

# Function to draw a smaller rectangle divided into boxes
def draw_small_vector(ax, filled_boxes, total_boxes):
    for i in range(total_boxes):
        color = colors[0] if i < filled_boxes else colors[1]
        rect = plt.Rectangle((i, 0), 1, 1, edgecolor='black', facecolor=color)
        ax.add_patch(rect)
    ax.set_xlim(0, total_boxes)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')

# Loop over the rows and columns to plot the rectangles
for row in range(16):
    for col in range(5):
        if col == 0:
            # The first column always has all boxes filled (4 red)
            filled_boxes = 4
        elif col == 1:
            # The second rectangle progressively adds more red boxes in all rows
            filled_boxes = min((row % 4) + 1, 4)
        elif col == 2 and row >= 4:
            # The third rectangle updates together with the second in the next 4 rows (5-8)
            filled_boxes = min((row % 4) + 1, 4) if row >= 4 else 0
        elif col == 3 and row >= 8:
            # The fourth rectangle updates together with the second and third in rows 9-12
            filled_boxes = min((row % 4) + 1, 4) if row >= 8 else 0
        elif col == 4 and row >= 12:
            # The fifth rectangle updates together with the others in the final rows (13-16)
            filled_boxes = min((row % 4) + 1, 4) if row >= 12 else 0
        else:
            # Other rectangles are gray if they haven't started replication yet
            filled_boxes = 0
        draw_small_vector(axs[row, col], filled_boxes, total_boxes)

# Adjust layout
plt.tight_layout()
plt.show()
